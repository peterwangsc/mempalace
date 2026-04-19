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

STOP_BLOCK_REASON = (
    "AUTO-SAVE checkpoint (MemPalace). Save this session's key content:\n"
    "1. mempalace_diary_write — AAAK-compressed session summary\n"
    "2. mempalace_add_drawer — verbatim quotes, decisions, code snippets\n"
    "3. mempalace_kg_add — entity relationships (optional)\n"
    "Do NOT write to Claude Code's native auto-memory (.md files). "
    "Continue conversation after saving."
)

PRECOMPACT_BLOCK_REASON = (
    "COMPACTION IMMINENT (MemPalace). Save ALL session content before context is lost:\n"
    "1. mempalace_diary_write — thorough AAAK-compressed session summary\n"
    "2. mempalace_add_drawer — ALL verbatim quotes, decisions, code, context\n"
    "3. mempalace_kg_add — entity relationships (optional)\n"
    "Be thorough \u2014 after compaction, detailed context will be lost. "
    "Do NOT write to Claude Code's native auto-memory (.md files). "
    "Save everything to MemPalace, then allow compaction to proceed."
)

# SessionEnd timeout in Claude Code is hard-capped at 1500ms by default.
# Anything heavier MUST be detached so it survives parent shutdown.
SESSION_END_STUB_WING = "wing_session_stub"


def _sanitize_session_id(session_id: str) -> str:
    """Only allow alnum, dash, underscore to prevent path traversal."""
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "", session_id)
    return sanitized or "unknown"


def _validate_transcript_path(transcript_path: str):
    """Resolve and validate a transcript path; return Path or None.

    Required for _maybe_auto_ingest's cursor mining: we must reject
    anything that doesn't look like a transcript before spawning a
    mine subprocess on its parent directory.

    Accepts:
      - .jsonl or .json extension only
      - no '..' traversal components in the original input
    """
    if not transcript_path:
        return None
    path = Path(transcript_path).expanduser().resolve()
    if path.suffix not in (".jsonl", ".json"):
        return None
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
    path = Path(transcript_path).expanduser()
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


def _log(message: str):
    """Append to hook state log file."""
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        log_path = STATE_DIR / "hook.log"
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(log_path, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
    except OSError:
        pass


def _output(data: dict):
    """Print JSON to stdout with consistent formatting (pretty-printed)."""
    print(json.dumps(data, indent=2, ensure_ascii=False))


def _maybe_auto_ingest(transcript_path: str = "", sync: bool = False):
    """Auto-ingest the active transcript and/or a configured directory.

    Two ingest sources, evaluated independently:
      1. The active session transcript (if `transcript_path` is a valid
         file). Uses `--mode convos --cursor` so only newly-appended
         lines are processed — matches Claude Code / Codex CLI's
         append-only JSONL semantics.
      2. The MEMPAL_DIR env var (if set and points at a directory).
         Runs the existing full mine on that directory. This preserves
         the original behavior for users who configured it explicitly.

    sync=False (default, used by stop hook): run in background via Popen.
    sync=True (used by precompact hook): block until done so memories
        land before context compaction destroys them.
    """
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    log_path = STATE_DIR / "hook.log"

    def _run(cmd):
        try:
            with open(log_path, "a") as log_f:
                if sync:
                    subprocess.run(cmd, stdout=log_f, stderr=log_f, timeout=60)
                else:
                    subprocess.Popen(cmd, stdout=log_f, stderr=log_f)
        except (OSError, subprocess.TimeoutExpired):
            pass

    # 1. Active transcript file → cursor-aware convo mine on its parent dir.
    #    The cursor mine is correctness-safe for non-Claude-Code files
    #    (it falls back internally), so passing the dir is fine even
    #    when other JSONL files live alongside.
    path = _validate_transcript_path(transcript_path)
    if path is not None and path.is_file():
        parent = str(path.parent)
        _run(
            [
                sys.executable,
                "-m",
                "mempalace",
                "mine",
                parent,
                "--mode",
                "convos",
                "--cursor",
            ]
        )

    # 2. MEMPAL_DIR fallback (preserved for backwards compatibility).
    mempal_dir = os.environ.get("MEMPAL_DIR", "")
    if mempal_dir and os.path.isdir(mempal_dir):
        _run([sys.executable, "-m", "mempalace", "mine", mempal_dir])


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


def _read_session_endpoints(transcript_path: str, max_chars: int = 400):
    """Pull (first_user_msg, last_user_msg, real_user_turn_count) from a
    Claude Code / Codex JSONL transcript. Bounded scan — must complete
    well under the 1.5s SessionEnd budget even on multi-MB transcripts.

    Skips command-injections (<command-message>) and tool_result-only
    user lines, mirroring _count_human_messages's filtering.
    """
    path = Path(transcript_path).expanduser()
    if not path.is_file():
        return None, None, 0
    first = None
    last = None
    count = 0
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = entry.get("message", {}) if isinstance(entry, dict) else {}
                text = None
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        if "<command-message>" in content:
                            continue
                        text = content
                    elif isinstance(content, list):
                        if all(
                            isinstance(b, dict) and b.get("type") == "tool_result"
                            for b in content
                        ):
                            continue
                        joined = " ".join(
                            b.get("text", "") for b in content if isinstance(b, dict)
                        )
                        if "<command-message>" in joined:
                            continue
                        text = joined
                elif isinstance(entry, dict) and entry.get("type") == "event_msg":
                    payload = entry.get("payload", {})
                    if isinstance(payload, dict) and payload.get("type") == "user_message":
                        m = payload.get("message", "")
                        if isinstance(m, str) and "<command-message>" not in m:
                            text = m
                if text is None:
                    continue
                count += 1
                snippet = text.strip()[:max_chars]
                if first is None:
                    first = snippet
                last = snippet
    except OSError:
        return None, None, 0
    return first, last, count


def _write_session_end_stub(
    session_id: str, transcript_path: str, reason: str, cwd: str
) -> bool:
    """Write a synthetic diary entry directly via the palace API — no MCP,
    no AI cooperation needed. Returns True on success.

    This is the load-bearing fix for the /clear data-loss case: even when
    no AI ever responds to a Stop or PreCompact block instruction, this
    stub guarantees the session leaves a discoverable marker behind.
    """
    try:
        from .palace import get_collection
        from .config import MempalaceConfig
    except Exception:
        return False

    try:
        palace_path = MempalaceConfig().palace_path
    except Exception:
        return False

    try:
        col = get_collection(palace_path, create=True)
    except Exception:
        return False
    if col is None:
        return False

    first_msg, last_msg, turn_count = _read_session_endpoints(transcript_path)
    now = datetime.now()
    body_lines = [
        f"AUTO-SESSION-END-STUB session={session_id} reason={reason}",
        f"timestamp={now.isoformat()}",
        f"cwd={cwd}",
        f"transcript_path={transcript_path}",
        f"real_user_turns={turn_count}",
    ]
    if first_msg:
        body_lines.append(f"first_user_msg: {first_msg}")
    if last_msg and last_msg != first_msg:
        body_lines.append(f"last_user_msg: {last_msg}")
    body_lines.append(
        "NOTE: synthetic stub written by mempal-session-end-hook. "
        "Drawers from the transcript are mined in a detached background "
        "process. A richer AAAK summary may supersede this entry if an "
        "agent later writes one for the same session."
    )
    body = "\n".join(body_lines)

    entry_id = (
        f"diary_{SESSION_END_STUB_WING}_{now.strftime('%Y%m%d_%H%M%S%f')}_"
        f"{re.sub(r'[^a-zA-Z0-9]', '', session_id)[:16]}"
    )
    try:
        col.upsert(
            ids=[entry_id],
            documents=[body],
            metadatas=[
                {
                    "wing": SESSION_END_STUB_WING,
                    "room": "diary",
                    "hall": "hall_diary",
                    "type": "session_end_stub",
                    "reason": reason,
                    "session_id": session_id,
                    "filed_at": now.isoformat(),
                    "date": now.strftime("%Y-%m-%d"),
                }
            ],
        )
        return True
    except Exception:
        return False


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


def hook_stop(data: dict, harness: str):
    """Stop hook: block every N messages for auto-save."""
    # Subagent stops fire the same Stop hook against the parent session's
    # transcript_path, so a single main-agent turn with N background
    # subagents produces N+1 firings at identical timestamps. That's
    # wasted CPU (mine_lock serializes them into identical work) and
    # noisy logs. Skip subagent stops — their content is already in the
    # parent transcript and gets ingested by the cursor mine when the
    # main agent stops.
    if data.get("agent_id"):
        _output({})
        return

    parsed = _parse_harness_input(data, harness)
    session_id = parsed["session_id"]
    stop_hook_active = parsed["stop_hook_active"]
    transcript_path = parsed["transcript_path"]

    # If already in a save cycle, let through (infinite-loop prevention)
    if str(stop_hook_active).lower() in ("true", "1", "yes"):
        _output({})
        return

    # Count human messages
    exchange_count = _count_human_messages(transcript_path)

    # Track last save point
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    last_save_file = STATE_DIR / f"{session_id}_last_save"
    last_save = 0
    if last_save_file.is_file():
        try:
            last_save = int(last_save_file.read_text().strip())
        except (ValueError, OSError):
            last_save = 0

    since_last = exchange_count - last_save

    _log(f"Session {session_id}: {exchange_count} exchanges, {since_last} since last save")

    if since_last >= SAVE_INTERVAL and exchange_count > 0:
        # Update last save point
        try:
            last_save_file.write_text(str(exchange_count), encoding="utf-8")
        except OSError:
            pass

        _log(f"TRIGGERING SAVE at exchange {exchange_count}")

        # Auto-ingest: cursor-mine the active transcript (background)
        # plus the MEMPAL_DIR fallback if configured. Hook never blocks
        # on this.
        _maybe_auto_ingest(transcript_path=transcript_path, sync=False)

        _output({"decision": "block", "reason": STOP_BLOCK_REASON})
    else:
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


def hook_precompact(data: dict, harness: str):
    """Precompact hook: always block with comprehensive save instruction."""
    parsed = _parse_harness_input(data, harness)
    session_id = parsed["session_id"]
    transcript_path = parsed["transcript_path"]

    _log(f"PRE-COMPACT triggered for session {session_id}")

    # Auto-ingest synchronously before compaction (so memories land
    # first). Cursor-mines the active transcript and, if MEMPAL_DIR is
    # set, runs the full mine on that directory too.
    _maybe_auto_ingest(transcript_path=transcript_path, sync=True)

    # Synthetic stub guarantees a discoverable marker even if the user
    # /clears immediately after /compact and the AI never responds to
    # this block instruction. The block message still asks the agent to
    # write a richer AAAK summary; the stub is just a floor.
    cwd = str(data.get("cwd", "") or "")
    if _write_session_end_stub(session_id, transcript_path, "precompact", cwd):
        _log(f"PRE-COMPACT stub written for session {session_id}")

    # Always block -- compaction = save everything
    _output({"decision": "block", "reason": PRECOMPACT_BLOCK_REASON})


def hook_session_end(data: dict, harness: str):
    """SessionEnd hook: fires on /clear, /logout, prompt-input-exit, etc.

    Hard 1.5s budget in Claude Code (overridable via
    CLAUDE_CODE_SESSIONEND_HOOKS_TIMEOUT_MS). We do two cheap things:

      1. Write a synthetic diary stub directly via the palace API —
         pure Python, well under the budget. This is the bare-minimum
         marker that survives /clear; without it, short sessions
         disappear entirely from MemPalace.
      2. Spawn the cursor mine in a *detached* process group so it
         survives Claude Code's teardown reaping. The mine itself can
         take seconds; it just runs after the session is gone.

    SessionEnd never blocks — there's no AI left to respond.
    """
    parsed = _parse_harness_input(data, harness)
    session_id = parsed["session_id"]
    transcript_path = parsed["transcript_path"]
    reason = str(data.get("reason", "") or "unknown")

    _log(f"SESSION-END triggered for session {session_id} reason={reason}")

    cwd = str(data.get("cwd", "") or "")
    if _write_session_end_stub(session_id, transcript_path, reason, cwd):
        _log(f"SESSION-END stub written for session {session_id}")

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
