"""
Hook logic for MemPalace — Python implementation of session-start, stop, and precompact hooks.

Reads JSON from stdin, outputs JSON to stdout.
Supported hooks: session-start, stop, precompact
Supported harnesses: claude-code, codex (extensible to cursor, gemini, etc.)
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SAVE_INTERVAL = 15
STATE_DIR = Path.home() / ".mempalace" / "hook_state"


def _mempalace_python() -> str:
    """Return the python interpreter that has mempalace installed.

    When hooks are invoked by Claude Code, sys.executable may be the system
    python which lacks chromadb and other deps.  Resolution order:
    1. MEMPALACE_PYTHON env var (explicit override)
    2. Venv python from package install path
    3. Editable install: venv/ sibling to mempalace/
    4. sys.executable fallback
    """
    # Honor explicit override (used by shell hook wrappers)
    env_python = os.environ.get("MEMPALACE_PYTHON", "")
    if env_python and os.path.isfile(env_python) and os.access(env_python, os.X_OK):
        return env_python
    # This file lives at <venv>/lib/pythonX.Y/site-packages/mempalace/hooks_cli.py
    # or <project>/mempalace/hooks_cli.py (editable install).
    venv_bin = Path(__file__).resolve().parents[3] / "bin" / "python"
    if venv_bin.is_file():
        return str(venv_bin)
    # Editable install: assumes project root has a venv/ sibling to mempalace/
    project_venv = Path(__file__).resolve().parents[1] / "venv" / "bin" / "python"
    if project_venv.is_file():
        return str(project_venv)
    return sys.executable


STOP_BLOCK_REASON = (
    "AUTO-SAVE checkpoint (MemPalace). Save this session's key content:\n"
    "1. mempalace_diary_write — session summary (what was discussed, "
    "key decisions, current state of work)\n"
    "2. mempalace_add_drawer — verbatim quotes, decisions, code snippets "
    "(place in appropriate wing and room)\n"
    "3. mempalace_kg_add — entity relationships (optional)\n"
    "For THIS save, use MemPalace MCP tools only (not auto-memory .md files). "
    "Use verbatim quotes where possible. Continue conversation after saving."
)

PRECOMPACT_BLOCK_REASON = (
    "COMPACTION IMMINENT (MemPalace). Save ALL session content before context is lost:\n"
    "1. mempalace_diary_write — thorough session summary\n"
    "2. mempalace_add_drawer — ALL verbatim quotes, decisions, code, context "
    "(place each in appropriate wing and room)\n"
    "3. mempalace_kg_add — entity relationships (optional)\n"
    "For THIS save, use MemPalace MCP tools only (not auto-memory .md files). "
    "Be thorough — after compaction this is all that survives. "
    "Save everything to MemPalace, then allow compaction to proceed."
)

# SessionEnd timeout in Claude Code is hard-capped at 1500ms by default.
# Anything heavier MUST be detached so it survives parent shutdown.


def _sanitize_session_id(session_id: str) -> str:
    """Only allow alnum, dash, underscore to prevent path traversal."""
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "", session_id)
    return sanitized or "unknown"


def _validate_transcript_path(transcript_path: str) -> Path:
    """Validate and resolve a transcript path, rejecting paths outside expected roots.

    Returns a resolved Path if valid, or None if the path should be rejected.
    Accepted paths must:
    - Have a .jsonl or .json extension
    - Not contain '..' after resolution (path traversal prevention)
    """
    if not transcript_path:
        return None
    path = Path(transcript_path).expanduser().resolve()
    if path.suffix not in (".jsonl", ".json"):
        return None
    # Reject if the original input contained '..' traversal components
    if ".." in Path(transcript_path).parts:
        return None
    return path


def _count_human_messages(transcript_path: str) -> int:
    """Count user-initiated messages in a JSONL transcript.

    Skips:
      - <command-message> slash-command injections
      - tool_result-only user lines (Claude Code represents tool output
        as role=user with content=[{type:"tool_result"...}]; these are
        assistant-initiated tool calls, not real user input)

    The counter drives SAVE_INTERVAL — so "message" must mean "the user
    typed at Claude," not "a user-role line appeared in the transcript."
    Tool-heavy turns previously inflated this count 10–20x and caused
    saves to fire ~every 1-2 real user turns in investigation-heavy
    sessions.
    """
    path = _validate_transcript_path(transcript_path)
    if path is None:
        if transcript_path:
            _log(f"WARNING: transcript_path rejected by validator: {transcript_path!r}")
        return 0
    if not path.is_file():
        return 0
    count = 0
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    msg = entry.get("message", {})
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            if "<command-message>" in content:
                                continue
                        elif isinstance(content, list):
                            # Tool-results-only lines aren't user input —
                            # they're the transcript echo of an assistant
                            # tool call. Skip entirely.
                            if all(
                                isinstance(b, dict) and b.get("type") == "tool_result"
                                for b in content
                            ):
                                continue
                            text = " ".join(
                                b.get("text", "") for b in content if isinstance(b, dict)
                            )
                            if "<command-message>" in text:
                                continue
                        count += 1
                    # Also handle Codex CLI transcript format
                    # {"type": "event_msg", "payload": {"type": "user_message", "message": "..."}}
                    elif entry.get("type") == "event_msg":
                        payload = entry.get("payload", {})
                        if isinstance(payload, dict) and payload.get("type") == "user_message":
                            msg_text = payload.get("message", "")
                            if isinstance(msg_text, str) and "<command-message>" not in msg_text:
                                count += 1
                except (json.JSONDecodeError, AttributeError):
                    pass
    except OSError:
        return 0
    return count


def _last_stop_mine_count_path(session_id: str) -> Path:
    return STATE_DIR / f"{_sanitize_session_id(session_id)}.last_stop_mine_count"


def _read_last_stop_mine_count(session_id: str) -> int:
    try:
        return int(_last_stop_mine_count_path(session_id).read_text().strip())
    except (OSError, ValueError):
        return 0


def _write_last_stop_mine_count(session_id: str, count: int) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        path = _last_stop_mine_count_path(session_id)
        path.write_text(str(count), encoding="utf-8")
        try:
            path.chmod(0o600)
        except (OSError, NotImplementedError):
            pass
    except OSError:
        pass


_state_dir_initialized = False


def _log(message: str):
    """Append to hook state log file."""
    global _state_dir_initialized
    try:
        if not _state_dir_initialized:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            try:
                STATE_DIR.chmod(0o700)
            except (OSError, NotImplementedError):
                pass
            _state_dir_initialized = True
        log_path = STATE_DIR / "hook.log"
        is_new = not log_path.exists()
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(log_path, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
        if is_new:
            try:
                log_path.chmod(0o600)
            except (OSError, NotImplementedError):
                pass
    except OSError:
        pass


def _output(data: dict):
    """Print JSON to stdout without importing modules that may redirect streams.

    If mempalace.mcp_server is already loaded, reuse its saved real stdout fd.
    Otherwise, write directly to fd 1 so hook responses still go to stdout even
    if sys.stdout has been redirected elsewhere.
    """
    payload = (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")

    real_stdout_fd: int | None = None
    mcp_mod = sys.modules.get("mempalace.mcp_server") or sys.modules.get(
        f"{__package__}.mcp_server" if __package__ else "mcp_server"
    )
    if mcp_mod is not None:
        real_stdout_fd = getattr(mcp_mod, "_REAL_STDOUT_FD", None)

    fd = real_stdout_fd if real_stdout_fd is not None else 1
    offset = 0
    try:
        while offset < len(payload):
            try:
                offset += os.write(fd, payload[offset:])
            except InterruptedError:
                continue
        return
    except OSError:
        pass

    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def _ingest_transcript(transcript_path: str) -> bool:
    """Mine a Claude Code session transcript into the palace as a conversation."""
    path = Path(transcript_path).expanduser()
    if not path.is_file() or path.stat().st_size < 100:
        return False

    from .config import MempalaceConfig

    try:
        MempalaceConfig()  # validate config loads
    except Exception:
        return False

    try:
        log_path = STATE_DIR / "hook.log"
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as log_f:
            subprocess.Popen(
                [
                    _mempalace_python(),
                    "-m",
                    "mempalace",
                    "mine",
                    str(path.parent),
                    "--mode",
                    "convos",
                    # No --wing: with the per-subdir wing derivation
                    # (66ee2e5), mining ~/.claude/projects/-Users-X-Y/
                    # auto-derives wing=_users_x_y, matching the
                    # canonical wing names already in the palace.
                    # Hardcoding --wing sessions would create a parallel
                    # generic wing instead of routing to the project wing.
                    #
                    # Append-only JSONL: only embed newly-appended bytes,
                    # not the full file each hook fire. Falls back to
                    # full-mine internally for non-eligible files.
                    "--cursor",
                ],
                stdout=log_f,
                stderr=log_f,
            )
        _log(f"Transcript ingest started (cursor mode): {path.name}")
        return True
    except OSError:
        return False


def _spawn_detached_cursor_mine(transcript_path: str) -> None:
    """Spawn `mempalace mine --cursor` in a detached session that
    survives parent shutdown. SessionEnd's 1.5s budget is too small
    to wait synchronously, and a normal Popen dies when Claude Code
    teardown reaps its process group.
    """
    path = _validate_transcript_path(transcript_path)
    if path is None or not path.is_file():
        return
    parent = str(path.parent)
    log_path = STATE_DIR / "hook.log"
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as log_f:
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "mempalace",
                    "mine",
                    parent,
                    "--mode",
                    "convos",
                    "--cursor",
                ],
                stdout=log_f,
                stderr=log_f,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
    except OSError:
        pass


SUPPORTED_HARNESSES = {"claude-code", "codex"}


def _parse_harness_input(data: dict, harness: str) -> dict:
    """Parse stdin JSON according to the harness type."""
    if harness not in SUPPORTED_HARNESSES:
        print(f"Unknown harness: {harness}", file=sys.stderr)
        sys.exit(1)
    return {
        "session_id": _sanitize_session_id(str(data.get("session_id", "unknown"))),
        "stop_hook_active": data.get("stop_hook_active", False),
        "transcript_path": str(data.get("transcript_path", "")),
    }


def _wing_from_transcript_path(transcript_path: str) -> str:
    """Derive the canonical wing for a transcript path.

    Wing convention: the wing for a transcript is the same wing the
    convo miner would derive when given that transcript's parent
    directory — i.e. ``normalize_wing_name(parent.name)``. This keeps
    diary checkpoints, SessionEnd stubs, and verbatim drawer ingest
    all routed to the SAME wing for a given project.

    For Claude Code transcripts at
        ~/.claude/projects/-Users-<user>-code-<project>/<session>.jsonl
    the parent dir name is ``-Users-<user>-code-<project>`` which
    normalizes to ``_users_<user>_code_<project>`` — exactly what
    ``mine_convos`` produces.

    Falls back to ``_sessions`` (canonical-prefixed, NOT ``wing_sessions``)
    only when no transcript path is supplied. Empty path → empty fallback
    that callers can reject. Never invents a new ``wing_*`` namespace —
    every legitimate wing on disk ultimately derives from
    ``normalize_wing_name`` so that the same project name produces the
    same wing across CLI mining, hook ingest, and synthetic stubs.
    """
    if not transcript_path:
        return "_sessions"
    try:
        from .config import normalize_wing_name
    except Exception:
        return "_sessions"
    parent = Path(transcript_path).expanduser().parent
    name = parent.name
    if not name:
        return "_sessions"
    return normalize_wing_name(name)


def hook_stop(data: dict, harness: str):
    """Stop hook: periodically spawn an incremental cursor mine of the transcript.

    The hook fires whenever the assistant stops, but mining is gated by
    SAVE_INTERVAL real user messages per session. The spawned
    `mempalace mine --cursor` subprocess writes through the locked
    ChromaCollection wrapper, so it serializes against every other
    palace writer (other sessions' MCP servers, other hook mines) via
    the cross-process `palace_write_lock`. Concurrent fires against the
    same transcript additionally coordinate through the per-file
    `mine_lock`.

    Subagent stops fire the same Stop hook against the parent session's
    transcript; their content is already in the parent transcript and
    gets ingested when the main agent stops. Skip them.
    """
    if data.get("agent_id"):
        _output({})
        return

    parsed = _parse_harness_input(data, harness)
    session_id = parsed["session_id"]
    transcript_path = parsed["transcript_path"]

    if transcript_path:
        user_message_count = _count_human_messages(transcript_path)
        last_mined_count = _read_last_stop_mine_count(session_id)
        if user_message_count > 0 and (
            user_message_count < last_mined_count
            or user_message_count - last_mined_count >= SAVE_INTERVAL
        ):
            _log(
                f"STOP threshold reached for session {session_id}: "
                f"{user_message_count} user messages ({user_message_count - last_mined_count} since last mine)"
            )
            if _ingest_transcript(transcript_path):
                _write_last_stop_mine_count(session_id, user_message_count)
        else:
            _log(
                f"STOP threshold not reached for session {session_id}: "
                f"{user_message_count} user messages ({user_message_count - last_mined_count} since last mine)"
            )

    _output({})


def hook_session_start(data: dict, harness: str):
    """Session start hook: initialize session tracking state."""
    parsed = _parse_harness_input(data, harness)
    session_id = parsed["session_id"]

    _log(f"SESSION START for session {session_id}")

    # Initialize session state directory
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    # Pass through — no blocking on session start
    _output({})


PRECOMPACT_CUSTOM_INSTRUCTIONS = """\
The full verbatim transcript of this session has already been mined into
MemPalace (drawers + closets, routed to the canonical project wing). A
multi-paragraph prose summary is therefore redundant — anything you would
say can be retrieved verbatim by searching the palace.

Override the default summary. Output ONLY a list of 3-5 mempalace_search
queries that, run in a fresh session, would let the next agent recover
the substance of THIS conversation. Choose queries that:
  - Use distinctive, project-specific terms (file names, symbols, error
    strings, named decisions) — not generic words like "fix" or "code".
  - Cover the major threads of the conversation, not just the most
    recent topic.
  - Span the actual scope discussed: what was attempted, what failed,
    what landed, what remains.

Format the output as a markdown bulleted list, each line a single search
query in backticks. No prose, no preamble, no closing remarks. Just the
queries. Example shape (NOT the content):

  - `<distinctive query 1>`
  - `<distinctive query 2>`
  - `<distinctive query 3>`
"""


def hook_precompact(data: dict, harness: str):
    """Precompact hook: spawn an incremental cursor mine of the transcript,
    then override compaction's default summary prompt with one that emits
    mempalace_search recovery queries (the verbatim is already in the
    palace, so a prose summary would be redundant).

    Single-purpose by design (personal-v3): only the spawned cursor mine
    writes to chromadb. The previous design also did MEMPAL_DIR sync mine
    + a synthetic SessionEnd stub upsert — those produced parallel writes
    that raced the spawned miner and corrupted HNSW segments.
    """
    parsed = _parse_harness_input(data, harness)
    session_id = parsed["session_id"]
    transcript_path = parsed["transcript_path"]

    _log(f"PRE-COMPACT triggered for session {session_id}")

    if transcript_path:
        _ingest_transcript(transcript_path)

    _output({"newCustomInstructions": PRECOMPACT_CUSTOM_INSTRUCTIONS})


def hook_session_end(data: dict, harness: str):
    """SessionEnd hook: fires on /clear, /logout, prompt-input-exit, etc.

    Hard 1.5s budget in Claude Code (overridable via
    CLAUDE_CODE_SESSIONEND_HOOKS_TIMEOUT_MS). Single-purpose by design
    (personal-v3): spawn the cursor mine in a *detached* process group
    so it survives Claude Code's teardown reaping, and return.

    The previous design also wrote a synthetic stub directly via the
    palace API — that's a parallel chromadb write that raced the
    detached miner and corrupted HNSW segments.

    SessionEnd never blocks — there's no AI left to respond.
    """
    parsed = _parse_harness_input(data, harness)
    session_id = parsed["session_id"]
    transcript_path = parsed["transcript_path"]
    reason = str(data.get("reason", "") or "unknown")

    _log(f"SESSION-END triggered for session {session_id} reason={reason}")

    _spawn_detached_cursor_mine(transcript_path)

    _output({})


def run_hook(hook_name: str, harness: str):
    """Main entry point: read stdin JSON, dispatch to hook handler."""
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        _log("WARNING: Failed to parse stdin JSON, proceeding with empty data")
        data = {}

    hooks = {
        "session-start": hook_session_start,
        "session-end": hook_session_end,
        "stop": hook_stop,
        "precompact": hook_precompact,
    }

    handler = hooks.get(hook_name)
    if handler is None:
        print(f"Unknown hook: {hook_name}", file=sys.stderr)
        sys.exit(1)

    handler(data, harness)
