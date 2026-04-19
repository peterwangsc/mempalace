#!/usr/bin/env python3
"""
convo_miner.py — Mine conversations into the palace.

Ingests chat exports (Claude Code, ChatGPT, Slack, plain text transcripts).
Normalizes format, chunks by exchange pair (Q+A = one unit), files to palace.

Same palace as project mining. Different ingest strategy.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional

from .normalize import normalize
from .palace import (
    NORMALIZE_VERSION,
    SKIP_DIRS,
    build_closet_lines,
    file_already_mined,
    get_closets_collection,
    get_collection,
    mine_lock,
    purge_file_closets,
    upsert_closet_lines,
)


# Cached hall keywords — avoids re-reading config per drawer
_HALL_KEYWORDS_CACHE = None


def _detect_hall_cached(content: str) -> str:
    """Route content to a hall using cached keywords. Same logic as miner.detect_hall."""
    global _HALL_KEYWORDS_CACHE
    if _HALL_KEYWORDS_CACHE is None:
        from .config import MempalaceConfig

        _HALL_KEYWORDS_CACHE = MempalaceConfig().hall_keywords
    content_lower = content[:3000].lower()
    scores = {}
    for hall, keywords in _HALL_KEYWORDS_CACHE.items():
        score = sum(1 for kw in keywords if kw in content_lower)
        if score > 0:
            scores[hall] = score
    return max(scores, key=scores.get) if scores else "general"


# File types that might contain conversations
CONVO_EXTENSIONS = {
    ".txt",
    ".md",
    ".json",
    ".jsonl",
}

MIN_CHUNK_SIZE = 30
CHUNK_SIZE = 800  # chars per drawer — align with miner.py
DRAWER_UPSERT_BATCH_SIZE = 1000
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB — skip files larger than this.
# Matches miner.py at 500 MB. Long Claude Code sessions, multi-year
# ChatGPT exports, and lifetime Slack dumps routinely exceed 10 MB; the
# cap at that level silently dropped them with `continue`. Per-drawer
# size is bounded by CHUNK_SIZE, but larger source files still produce
# more drawers and therefore more embedding/storage work — and content
# is normalized and loaded fully into memory before chunking, so memory
# use also scales with source size.


def _register_file(collection, source_file: str, wing: str, agent: str):
    """Write a sentinel so file_already_mined() returns True for 0-chunk files.

    Without this, files that normalize to nothing or produce zero chunks are
    re-read and re-processed on every mine run because nothing was written to
    ChromaDB on the first pass.
    """
    sentinel_id = f"_reg_{hashlib.sha256(source_file.encode()).hexdigest()[:24]}"
    collection.upsert(
        documents=[f"[registry] {source_file}"],
        ids=[sentinel_id],
        metadatas=[
            {
                "wing": wing,
                "room": "_registry",
                "source_file": source_file,
                "added_by": agent,
                "filed_at": datetime.now().isoformat(),
                "ingest_mode": "registry",
                "normalize_version": NORMALIZE_VERSION,
            }
        ],
    )


# =============================================================================
# CURSOR PRIMITIVES — append-only ingest for Claude Code / Codex JSONL
# =============================================================================
#
# Claude Code and Codex CLI write JSONL transcripts that are *strictly
# append-only* (verified at the source — fsAppendFile only, no in-place
# mutation, even under tool-use, sub-agents, and compaction). We exploit
# this to skip re-embedding the entire file every time the save hook
# fires; only the byte tail past a stored cursor is processed.
#
# Two correctness traps:
#   1. normalize._try_claude_code_jsonl merges consecutive assistant
#      messages and merges tool-result-only user messages into the
#      preceding assistant message. So a naive byte cursor would diverge
#      from full re-normalization at the trailing message. The cursor
#      stops at a *safe boundary* — the byte offset just past the last
#      JSONL line that won't be merged into by future appended lines.
#   2. Crash safety: the cursor is written LAST in the mine flow, after
#      all drawers commit. If we crash before the cursor write, the next
#      run reprocesses the same range; content-addressed drawer IDs make
#      the upserts idempotent (no duplicates, no missing data).

# Formats whose JSONL is append-stable AND whose normalizer can be
# safely resumed from a byte cursor.
_CURSOR_ELIGIBLE_SUFFIXES = {".jsonl"}


def _is_cursor_eligible(filepath: Path) -> bool:
    """Cheap pre-check: does this file look like Claude Code or Codex JSONL?

    We sniff a few lines to confirm the format rather than trusting the
    extension alone — a `.jsonl` file could be anything. Conservative:
    any parse failure or unrecognized shape returns False, falling
    through to the existing full-mine path.
    """
    if filepath.suffix.lower() not in _CURSOR_ELIGIBLE_SUFFIXES:
        return False
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for _ in range(20):
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    return False
                if not isinstance(entry, dict):
                    return False
                # Claude Code: {"type": "user"|"assistant", "message": {...}}
                if entry.get("type") in ("user", "human", "assistant") and isinstance(
                    entry.get("message"), dict
                ):
                    return True
                # Codex CLI: {"type": "session_meta"} or {"type": "event_msg", ...}
                if entry.get("type") in ("session_meta", "event_msg"):
                    return True
    except OSError:
        return False
    return False


def _find_safe_boundary(filepath: Path) -> int:
    """Return the byte offset just after the last 'safe' JSONL line.

    A safe line is one that, if more lines are appended, won't be merged
    into by the normalizer. For Claude Code JSONL, that means a
    user-role line whose content is NOT a list-of-tool-results-only —
    because tool-result user lines and consecutive assistant lines both
    merge upward into the prior assistant message. For Codex JSONL,
    every event_msg is independent (no merging), so the last complete
    line is safe.

    Returns 0 if no safe boundary is found (cursor mode should not
    advance). The returned offset always points at a newline boundary
    in the file — partial trailing lines are never included.
    """
    safe_offset = 0
    pos = 0
    is_codex_format = False
    try:
        with open(filepath, "rb") as f:
            for raw in f:
                pos_after = pos + len(raw)
                # Lines must end with \n to be considered complete; a
                # trailing partial line (no newline) is in-flight and not safe.
                if not raw.endswith(b"\n"):
                    break
                stripped = raw.strip()
                if not stripped:
                    pos = pos_after
                    continue
                try:
                    entry = json.loads(stripped.decode("utf-8", errors="replace"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pos = pos_after
                    continue
                if not isinstance(entry, dict):
                    pos = pos_after
                    continue

                etype = entry.get("type", "")

                # Codex format: every event_msg is an independent message;
                # the boundary advances on every complete line.
                if etype in ("session_meta", "event_msg"):
                    is_codex_format = True
                    safe_offset = pos_after
                    pos = pos_after
                    continue

                # Claude Code format: only advance on a user message that
                # is NOT tool-results-only.
                if etype in ("user", "human"):
                    msg = entry.get("message", {})
                    if isinstance(msg, dict):
                        content = msg.get("content", "")
                        is_tool_only = isinstance(content, list) and all(
                            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
                        )
                        if not is_tool_only:
                            safe_offset = pos_after
                pos = pos_after
    except OSError:
        return 0

    # Defensive: if neither format was recognized, decline to advance.
    if safe_offset == 0 and not is_codex_format:
        return 0
    return safe_offset


def _read_cursor(collection, source_file: str) -> int:
    """Read the stored byte cursor for source_file. Returns 0 if absent.

    Cursor lives on the same `_reg_<sha>` sentinel record that already
    tracks file_already_mined() state. Any read error returns 0
    (full mine).
    """
    try:
        sentinel_id = f"_reg_{hashlib.sha256(source_file.encode()).hexdigest()[:24]}"
        result = collection.get(ids=[sentinel_id])
        metas = result.get("metadatas") or []
        if not metas:
            return 0
        cursor = metas[0].get("byte_cursor", 0)
        if not isinstance(cursor, (int, float)):
            return 0
        return int(cursor)
    except Exception:
        return 0


def _write_cursor(collection, source_file: str, wing: str, agent: str, byte_offset: int) -> None:
    """Write the byte cursor to the sentinel. Called LAST in the mine flow.

    Crash before this point → next run re-mines the same range →
    content-hash drawer IDs make the upserts idempotent → no data loss.
    """
    sentinel_id = f"_reg_{hashlib.sha256(source_file.encode()).hexdigest()[:24]}"
    try:
        collection.upsert(
            documents=[f"[registry] {source_file}"],
            ids=[sentinel_id],
            metadatas=[
                {
                    "wing": wing,
                    "room": "_registry",
                    "source_file": source_file,
                    "added_by": agent,
                    "filed_at": datetime.now().isoformat(),
                    "ingest_mode": "registry",
                    "normalize_version": NORMALIZE_VERSION,
                    "byte_cursor": int(byte_offset),
                }
            ],
        )
    except Exception:
        # Sentinel write failure is non-fatal — drawers are committed.
        pass


# =============================================================================
# CHUNKING — exchange pairs for conversations
# =============================================================================


def chunk_exchanges(content: str) -> list:
    """
    Chunk by exchange pair: one > turn + AI response = one unit.
    Falls back to paragraph chunking if no > markers.
    """
    lines = content.split("\n")
    quote_lines = sum(1 for line in lines if line.strip().startswith(">"))

    if quote_lines >= 3:
        return _chunk_by_exchange(lines)
    else:
        return _chunk_by_paragraph(content)


def _chunk_by_exchange(lines: list) -> list:
    """One user turn (>) + the AI response that follows = one or more chunks.

    The full AI response is preserved verbatim.  When the combined
    user-turn + response exceeds CHUNK_SIZE the response is split across
    consecutive drawers so nothing is silently discarded.
    """
    chunks = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if line.strip().startswith(">"):
            user_turn = line.strip()
            i += 1

            ai_lines = []
            while i < len(lines):
                next_line = lines[i]
                if next_line.strip().startswith(">") or next_line.strip().startswith("---"):
                    break
                if next_line.strip():
                    ai_lines.append(next_line.strip())
                i += 1

            ai_response = " ".join(ai_lines)
            content = f"{user_turn}\n{ai_response}" if ai_response else user_turn

            # Split into multiple drawers when the exchange exceeds CHUNK_SIZE
            if len(content) > CHUNK_SIZE:
                # First chunk: user turn + as much response as fits
                first_part = content[:CHUNK_SIZE]
                if len(first_part.strip()) > MIN_CHUNK_SIZE:
                    chunks.append({"content": first_part, "chunk_index": len(chunks)})
                # Remaining response in CHUNK_SIZE-sized continuation drawers
                remainder = content[CHUNK_SIZE:]
                while remainder:
                    part = remainder[:CHUNK_SIZE]
                    remainder = remainder[CHUNK_SIZE:]
                    if len(part.strip()) > MIN_CHUNK_SIZE:
                        chunks.append({"content": part, "chunk_index": len(chunks)})
            elif len(content.strip()) > MIN_CHUNK_SIZE:
                chunks.append(
                    {
                        "content": content,
                        "chunk_index": len(chunks),
                    }
                )
        else:
            i += 1

    return chunks


def _chunk_by_paragraph(content: str) -> list:
    """Fallback: chunk by paragraph breaks."""
    chunks = []
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    # If no paragraph breaks and long content, chunk by line groups
    if len(paragraphs) <= 1 and content.count("\n") > 20:
        lines = content.split("\n")
        for i in range(0, len(lines), 25):
            group = "\n".join(lines[i : i + 25]).strip()
            if len(group) > MIN_CHUNK_SIZE:
                chunks.append({"content": group, "chunk_index": len(chunks)})
        return chunks

    for para in paragraphs:
        if len(para) > MIN_CHUNK_SIZE:
            chunks.append({"content": para, "chunk_index": len(chunks)})

    return chunks


# =============================================================================
# ROOM DETECTION — topic-based for conversations
# =============================================================================

TOPIC_KEYWORDS = {
    "technical": [
        "code",
        "python",
        "function",
        "bug",
        "error",
        "api",
        "database",
        "server",
        "deploy",
        "git",
        "test",
        "debug",
        "refactor",
    ],
    "architecture": [
        "architecture",
        "design",
        "pattern",
        "structure",
        "schema",
        "interface",
        "module",
        "component",
        "service",
        "layer",
    ],
    "planning": [
        "plan",
        "roadmap",
        "milestone",
        "deadline",
        "priority",
        "sprint",
        "backlog",
        "scope",
        "requirement",
        "spec",
    ],
    "decisions": [
        "decided",
        "chose",
        "picked",
        "switched",
        "migrated",
        "replaced",
        "trade-off",
        "alternative",
        "option",
        "approach",
    ],
    "problems": [
        "problem",
        "issue",
        "broken",
        "failed",
        "crash",
        "stuck",
        "workaround",
        "fix",
        "solved",
        "resolved",
    ],
}


def detect_convo_room(content: str) -> str:
    """Score conversation content against topic keywords."""
    content_lower = content[:3000].lower()
    scores = {}
    for room, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in content_lower)
        if score > 0:
            scores[room] = score
    if scores:
        return max(scores, key=scores.get)
    return "general"


# =============================================================================
# PALACE OPERATIONS
# =============================================================================


# =============================================================================
# SCAN FOR CONVERSATION FILES
# =============================================================================


def scan_convos(convo_dir: str) -> list:
    """Find all potential conversation files."""
    convo_path = Path(convo_dir).expanduser().resolve()
    files = []
    for root, dirs, filenames in os.walk(convo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for filename in filenames:
            if filename.endswith(".meta.json"):
                continue
            filepath = Path(root) / filename
            if filepath.suffix.lower() in CONVO_EXTENSIONS:
                # Skip symlinks and oversized files
                if filepath.is_symlink():
                    continue
                try:
                    if filepath.stat().st_size > MAX_FILE_SIZE:
                        continue
                except OSError:
                    continue
                files.append(filepath)
    return files


# =============================================================================
# MINE CONVERSATIONS
# =============================================================================


def _normalize_tail(filepath: Path, cursor: int, tail_bytes: bytes) -> str:
    """Normalize the tail bytes [cursor, safe_boundary) to transcript text.

    Returns:
        normalized transcript text, or
        "_advance_only" sentinel meaning "no chunkable content but
        cursor should still advance" (e.g. the synthetic input had no
        new user marker), or
        "" / None if normalization produced nothing usable.

    Reconstructs the in-memory state the full normalizer would have if
    it had read the whole file: the tool_use_map is rebuilt from prefix
    assistant lines so tool_result blocks in the tail render with the
    right tool name instead of "Unknown".
    """
    from .normalize import _try_claude_code_jsonl, _try_codex_jsonl, strip_noise

    tail_text = tail_bytes.decode("utf-8", errors="replace")

    # Codex format has no merging and no tool_use_map — normalize directly.
    normalized = _try_codex_jsonl(tail_text)
    if normalized is None:
        # Claude Code: synthesize prefix assistant lines + tail so the
        # normalizer rebuilds the tool_use_map from real prior context.
        with open(filepath, "rb") as f:
            prefix_bytes = f.read(cursor)
        prefix_text = prefix_bytes.decode("utf-8", errors="replace")

        prefix_assistant_lines = []
        for line in prefix_text.split("\n"):
            line_s = line.strip()
            if not line_s:
                continue
            try:
                e = json.loads(line_s)
            except json.JSONDecodeError:
                continue
            if isinstance(e, dict) and e.get("type") == "assistant":
                prefix_assistant_lines.append(line_s)

        synthetic = "\n".join(prefix_assistant_lines + [tail_text.strip()])
        normalized = _try_claude_code_jsonl(synthetic)
        if normalized:
            # Strip everything before the first user-turn marker that
            # came from the tail. If no user marker appears, the tail
            # had no new user turn — return the advance-only sentinel.
            first_user_marker = normalized.find("\n> ")
            if first_user_marker >= 0:
                normalized = normalized[first_user_marker + 1 :]
            elif not normalized.lstrip().startswith(">"):
                return "_advance_only"

    if normalized:
        normalized = strip_noise(normalized)
    return normalized or ""


def _rebuild_closets_for_file(
    drawers_col,
    closets_col,
    source_file: str,
    wing: str,
    room,
    agent: str,
    smart_ctx: Optional[dict] = None,
) -> None:
    """(Re)build the closet index lines for one source file from the
    drawers currently stored for it. Mirrors miner.py:624-648 for the
    convo path, which historically skipped closet building entirely.

    Queries chroma for the canonical drawer set, derives closets from
    joined content, then purge+upsert so a re-mine never leaves
    orphaned closets behind. Safe to call after both the full-mine
    path and the cursor-mine path: the cursor path appends new drawers
    incrementally, and rebuilding closets from the union picks up the
    new content automatically.

    Failures are swallowed — closets are a search rank-boost, not the
    source of truth. We never block drawer writes on closet trouble.
    """
    if drawers_col is None or closets_col is None:
        return
    try:
        result = drawers_col.get(
            where={"source_file": source_file},
            include=["documents", "metadatas"],
        )
    except Exception:
        return

    drawer_ids = result.get("ids", []) or []
    docs = result.get("documents", []) or []
    metas = result.get("metadatas", []) or []
    if not drawer_ids or not docs:
        return

    # Closet metadata wants a single representative room; general
    # extract mode varies room per chunk, so pick the most common one
    # observed in stored drawer metadata.
    room_for_closet = room
    if room_for_closet is None:
        from collections import Counter

        room_counter = Counter(m.get("room", "general") for m in metas if isinstance(m, dict))
        room_for_closet = room_counter.most_common(1)[0][0] if room_counter else "general"

    content = "\n\n".join(docs)
    lines = []
    ingest_mode = "convos"
    if smart_ctx and smart_ctx.get("idf_index") and smart_ctx.get("embedder") is not None:
        from .smart_closets import build_smart_closet_lines

        try:
            lines = build_smart_closet_lines(
                source_file,
                drawer_ids,
                content,
                wing,
                room_for_closet,
                idf_index=smart_ctx["idf_index"],
                embedder=smart_ctx["embedder"],
            )
            if lines:
                ingest_mode = "convos-smart"
        except Exception:
            lines = []
    if not lines:
        try:
            lines = build_closet_lines(source_file, drawer_ids, content, wing, room_for_closet)
        except Exception:
            return
    if not lines:
        return

    closet_id_base = (
        f"closet_{wing}_{room_for_closet}_"
        f"{hashlib.sha256(source_file.encode()).hexdigest()[:24]}"
    )
    closet_meta = {
        "wing": wing,
        "room": room_for_closet,
        "source_file": source_file,
        "drawer_count": len(drawer_ids),
        "filed_at": datetime.now().isoformat(),
        "normalize_version": NORMALIZE_VERSION,
        "ingest_mode": ingest_mode,
        "added_by": agent,
    }
    try:
        with mine_lock(source_file):
            purge_file_closets(closets_col, source_file)
            upsert_closet_lines(closets_col, closet_id_base, lines, closet_meta)
    except Exception:
        return


def _mine_with_cursor(
    collection,
    source_file: str,
    wing: str,
    agent: str,
    extract_mode: str,
    closets_col=None,
    smart_ctx: Optional[dict] = None,
) -> tuple:
    """Cursor-aware mine for an append-only JSONL source.

    Returns (drawers_added, room_counts_delta, status) where status is
    one of:
        "ok"       — cursor advanced, new drawers (possibly 0) appended
        "fallback" — cursor path declined; caller should run the
                     existing full-mine path
        "skipped"  — already-current; no work needed
    """
    filepath = Path(source_file)
    room_counts_delta: dict = defaultdict(int)

    # Sniff the format. Decline if it doesn't look like a known
    # append-only JSONL.
    if not _is_cursor_eligible(filepath):
        return 0, room_counts_delta, "fallback"

    safe_offset = _find_safe_boundary(filepath)
    if safe_offset <= 0:
        # No safe boundary yet (e.g. brand-new file with no completed
        # user turn). Fall through to full mine — it will produce a
        # normal first ingest, after which the cursor takes over.
        return 0, room_counts_delta, "fallback"

    with mine_lock(source_file):
        cursor = _read_cursor(collection, source_file)

        # Cursor sanity checks. Anything weird → fall back to full mine.
        try:
            file_size = os.path.getsize(filepath)
        except OSError:
            return 0, room_counts_delta, "fallback"
        if cursor < 0 or cursor > file_size:
            return 0, room_counts_delta, "fallback"
        # First-time mine for this file: defer to full mine so the
        # entire transcript gets ingested in one pass. The full-mine
        # path handles short transcripts and tool-loop-heavy sessions
        # correctly. Cursor mode then takes over for subsequent
        # incremental appends — the caller (mine_convos) writes the
        # cursor after the full mine completes successfully.
        if cursor == 0:
            return 0, room_counts_delta, "fallback"
        if cursor >= safe_offset:
            return 0, room_counts_delta, "skipped"

        # Read [cursor, safe_offset). Both are line-aligned by construction.
        try:
            with open(filepath, "rb") as f:
                f.seek(cursor)
                tail_bytes = f.read(safe_offset - cursor)
        except OSError:
            return 0, room_counts_delta, "fallback"

        if not tail_bytes.strip():
            _write_cursor(collection, source_file, wing, agent, safe_offset)
            return 0, room_counts_delta, "ok"

        try:
            normalized = _normalize_tail(filepath, cursor, tail_bytes)
        except Exception:
            return 0, room_counts_delta, "fallback"
        if normalized == "_advance_only":
            _write_cursor(collection, source_file, wing, agent, safe_offset)
            return 0, room_counts_delta, "ok"

        if not normalized or len(normalized.strip()) < MIN_CHUNK_SIZE:
            _write_cursor(collection, source_file, wing, agent, safe_offset)
            return 0, room_counts_delta, "ok"

        # Chunk + write. Reuses the same chunker the full path uses.
        if extract_mode == "general":
            from .general_extractor import extract_memories

            chunks = extract_memories(normalized)
            room = None
        else:
            chunks = chunk_exchanges(normalized)
            room = detect_convo_room(normalized)

        if not chunks:
            _write_cursor(collection, source_file, wing, agent, safe_offset)
            return 0, room_counts_delta, "ok"

        # Append drawers WITHOUT purging — content-addressed IDs make
        # this safe (re-runs upsert to the same ID; no duplicates).
        drawers_added = 0
        for chunk in chunks:
            chunk_room = chunk.get("memory_type", room) if extract_mode == "general" else room
            if extract_mode == "general":
                room_counts_delta[chunk_room] += 1
            drawer_id = (
                f"drawer_{wing}_{chunk_room}_"
                f"{hashlib.sha256((source_file + chunk['content']).encode()).hexdigest()[:24]}"
            )
            try:
                collection.upsert(
                    documents=[chunk["content"]],
                    ids=[drawer_id],
                    metadatas=[
                        {
                            "wing": wing,
                            "room": chunk_room,
                            "hall": _detect_hall_cached(chunk["content"]),
                            "source_file": source_file,
                            "chunk_index": chunk["chunk_index"],
                            "added_by": agent,
                            "filed_at": datetime.now().isoformat(),
                            "ingest_mode": "convos",
                            "extract_mode": extract_mode,
                            "normalize_version": NORMALIZE_VERSION,
                            "cursor_appended": True,
                        }
                    ],
                )
                drawers_added += 1
            except Exception as e:
                if "already exists" not in str(e).lower():
                    # Hard write failure — bail without advancing cursor
                    # so next run retries the same range.
                    return drawers_added, room_counts_delta, "fallback"

        # Cursor advances LAST. If anything above raised, we never reach
        # here and the next run re-processes the same tail (idempotent
        # via content-hash IDs).
        _write_cursor(collection, source_file, wing, agent, safe_offset)

    # Rebuild closets OUTSIDE mine_lock — _rebuild_closets_for_file
    # acquires its own mine_lock for the purge+upsert, and re-entrant
    # locking isn't guaranteed across backends.
    if drawers_added > 0:
        _rebuild_closets_for_file(
            collection,
            closets_col,
            source_file,
            wing,
            room if extract_mode != "general" else None,
            agent,
            smart_ctx=smart_ctx,
        )
    return drawers_added, room_counts_delta, "ok"


def _file_chunks_locked(
    collection,
    source_file,
    chunks,
    wing,
    room,
    agent,
    extract_mode,
    closets_col=None,
    smart_ctx: Optional[dict] = None,
):
    """Lock the source file, purge stale drawers, and upsert fresh chunks.

    Combines the per-file serialization that prevents concurrent agents from
    duplicating work (via mine_lock) with the normalize-version rebuild
    contract (purge-before-insert so pre-v2 drawers don't survive).

    Returns (drawers_added, room_counts_delta, skipped).
    """
    room_counts_delta: dict = defaultdict(int)
    drawers_added = 0
    with mine_lock(source_file):
        # Re-check after lock — another agent may have just finished this file
        # at the current schema. A stale-version hit here returns False, so we
        # still fall through to the purge+rebuild path below.
        if file_already_mined(collection, source_file):
            return 0, room_counts_delta, True

        # Purge stale drawers first. When the normalize schema bumps,
        # file_already_mined() returned False for pre-v2 drawers — clean
        # them out so the source doesn't end up with mixed old/new drawers.
        try:
            collection.delete(where={"source_file": source_file})
        except Exception:
            pass

        # Batch chunks into bounded upserts so large transcripts keep most of
        # the embedding speedup without one huge Chroma/SQLite request. Keep
        # one filed_at per source file so all transcript drawers share an
        # ingest timestamp. Drawer IDs are content-addressed (chunk content
        # hashed, NOT chunk_index) so re-mining the same file produces
        # idempotent upserts and cursor-based incremental ingest can append
        # new chunks without touching old ones — appending new exchanges to
        # a Claude Code transcript shifts no prior chunk's content, so prior
        # IDs remain stable across re-mines.
        filed_at = datetime.now().isoformat()
        for batch_start in range(0, len(chunks), DRAWER_UPSERT_BATCH_SIZE):
            batch_docs: list = []
            batch_ids: list = []
            batch_metas: list = []
            for chunk in chunks[batch_start : batch_start + DRAWER_UPSERT_BATCH_SIZE]:
                chunk_room = chunk.get("memory_type", room) if extract_mode == "general" else room
                if extract_mode == "general":
                    room_counts_delta[chunk_room] += 1
                drawer_id = f"drawer_{wing}_{chunk_room}_{hashlib.sha256((source_file + chunk['content']).encode()).hexdigest()[:24]}"
                batch_docs.append(chunk["content"])
                batch_ids.append(drawer_id)
                batch_metas.append(
                    {
                        "wing": wing,
                        "room": chunk_room,
                        "hall": _detect_hall_cached(chunk["content"]),
                        "source_file": source_file,
                        "chunk_index": chunk["chunk_index"],
                        "added_by": agent,
                        "filed_at": filed_at,
                        "ingest_mode": "convos",
                        "extract_mode": extract_mode,
                        "normalize_version": NORMALIZE_VERSION,
                    }
                )
            try:
                collection.upsert(
                    documents=batch_docs,
                    ids=batch_ids,
                    metadatas=batch_metas,
                )
                drawers_added += len(batch_docs)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    raise

    # Build closets OUTSIDE mine_lock — _rebuild_closets_for_file
    # acquires its own. Mirrors miner.py:624-648 for the convo path.
    if drawers_added > 0:
        _rebuild_closets_for_file(
            collection,
            closets_col,
            source_file,
            wing,
            room if extract_mode != "general" else None,
            agent,
            smart_ctx=smart_ctx,
        )
    return drawers_added, room_counts_delta, False


def _try_cursor_path(
    collection,
    source_file: str,
    wing: str,
    agent: str,
    extract_mode: str,
    closets_col=None,
    smart_ctx: Optional[dict] = None,
) -> tuple:
    """Attempt the cursor-aware mine path; absorb any unexpected failure.

    Returns (drawers_added, room_counts_delta, status) — same contract
    as _mine_with_cursor, except an exception from the inner call is
    caught and converted to status="fallback" so the main loop can drop
    into the full-mine path. Defensive on purpose: any bug in cursor
    logic must NEVER prevent the user's content from getting mined.
    """
    try:
        return _mine_with_cursor(
            collection,
            source_file,
            wing,
            agent,
            extract_mode,
            closets_col=closets_col,
            smart_ctx=smart_ctx,
        )
    except Exception:
        return 0, defaultdict(int), "fallback"


def _consume_cursor_result(
    drawers_added: int,
    room_delta: dict,
    status: str,
    filepath: Path,
    i: int,
    total_files: int,
    room_counts: dict,
) -> tuple:
    """Apply a cursor-path result to the running counters.

    Returns (handled, total_drawers_delta, files_skipped_delta) where
    handled=True means the caller should `continue` (cursor handled
    this file); handled=False means fall through to the full-mine path.
    """
    if status == "skipped":
        return True, 0, 1
    if status == "ok":
        for r, n in room_delta.items():
            room_counts[r] += n
        if drawers_added:
            print(f"  ✓ [{i:4}/{total_files}] {filepath.name[:50]:50} +{drawers_added} (cursor)")
        return True, drawers_added, 0
    return False, 0, 0


def _init_cursor_after_full_mine(
    collection, filepath: Path, source_file: str, wing: str, agent: str
) -> None:
    """Best-effort cursor initialization after a successful full mine.

    Called from mine_convos when cursor mode is enabled and the file is
    cursor-eligible, so subsequent mines can resume incrementally
    instead of re-processing the whole file. Failure here just leaves
    the cursor unset — the next mine falls through to full mine again
    (correct, just slower).
    """
    if not _is_cursor_eligible(filepath):
        return
    try:
        boundary = _find_safe_boundary(filepath)
        if boundary > 0:
            _write_cursor(collection, source_file, wing, agent, boundary)
    except Exception:
        pass


def _report_dry_run_chunks(filepath: Path, chunks: list, room, extract_mode: str, room_counts):
    """Print + tally the dry-run chunk preview for one file. Mutates room_counts."""
    if extract_mode == "general":
        from collections import Counter

        type_counts = Counter(c.get("memory_type", "general") for c in chunks)
        types_str = ", ".join(f"{t}:{n}" for t, n in type_counts.most_common())
        print(f"    [DRY RUN] {filepath.name} → {len(chunks)} memories ({types_str})")
        for c in chunks:
            room_counts[c.get("memory_type", "general")] += 1
    else:
        print(f"    [DRY RUN] {filepath.name} → room:{room} ({len(chunks)} drawers)")
        room_counts[room] += 1


def mine_convos(
    convo_dir: str,
    palace_path: str,
    wing: str = None,
    agent: str = "mempalace",
    limit: int = 0,
    dry_run: bool = False,
    extract_mode: str = "exchange",
    cursor: bool = False,
    smart_closets: bool = False,
):
    """Mine a directory of conversation files into the palace.

    extract_mode:
        "exchange" — default exchange-pair chunking (Q+A = one unit)
        "general"  — general extractor: decisions, preferences, milestones, problems, emotions

    cursor:
        False (default) — current behavior: file_already_mined() gates
                           re-mining on the assumption transcripts are
                           immutable.
        True            — for append-only sources (Claude Code JSONL,
                           Codex CLI JSONL), seek past a stored byte
                           cursor and only process newly-appended
                           content. Falls back to the full-mine path
                           for any source that isn't recognized as
                           append-stable, so it's safe to enable
                           globally.
    """

    convo_path = Path(convo_dir).expanduser().resolve()
    if not wing:
        from .config import normalize_wing_name

        wing = normalize_wing_name(convo_path.name)

    files = scan_convos(convo_dir)
    if limit > 0:
        files = files[:limit]

    print(f"\n{'=' * 55}")
    print("  MemPalace Mine — Conversations")
    print(f"{'=' * 55}")
    print(f"  Wing:    {wing}")
    print(f"  Source:  {convo_path}")
    print(f"  Files:   {len(files)}")
    print(f"  Palace:  {palace_path}")
    if dry_run:
        print("  DRY RUN — nothing will be filed")
    print(f"{'-' * 55}\n")

    collection = get_collection(palace_path) if not dry_run else None
    closets_col = get_closets_collection(palace_path) if not dry_run else None

    # Smart closets: load (or build) the corpus-wide IDF index and reach the
    # embedder through the collection adapter. `smart_ctx=None` keeps the
    # regex pipeline for anyone who doesn't want the extra cost/deps.
    smart_ctx = None
    if smart_closets and not dry_run and collection is not None:
        from .smart_closets import _embedder_from_drawers_col, get_or_build_idf_index

        embedder = _embedder_from_drawers_col(collection)
        if embedder is not None:
            idf_index = get_or_build_idf_index(palace_path, collection, progress=True)
            smart_ctx = {"idf_index": idf_index, "embedder": embedder}
            print(f"  Smart closets: n_docs={idf_index['n_docs']}, ngrams={len(idf_index['idf'])}")
        else:
            print("  Smart closets: embedder unavailable → falling back to regex")

    total_drawers = 0
    files_skipped = 0
    room_counts = defaultdict(int)

    for i, filepath in enumerate(files, 1):
        source_file = str(filepath)

        # Cursor-aware path for append-only JSONL sources (Claude Code,
        # Codex CLI). Dry-run is excluded so it never touches the
        # sentinel record. Any "fallback" status drops through to the
        # existing full-mine path.
        if cursor and not dry_run:
            drawers_added, room_delta, status = _try_cursor_path(
                collection,
                source_file,
                wing,
                agent,
                extract_mode,
                closets_col=closets_col,
                smart_ctx=smart_ctx,
            )
            handled, d_delta, s_delta = _consume_cursor_result(
                drawers_added, room_delta, status, filepath, i, len(files), room_counts
            )
            total_drawers += d_delta
            files_skipped += s_delta
            if handled:
                continue

        # Skip if already filed
        if not dry_run and file_already_mined(collection, source_file):
            files_skipped += 1
            continue

        # Normalize format
        try:
            content = normalize(str(filepath))
        except (OSError, ValueError):
            if not dry_run:
                _register_file(collection, source_file, wing, agent)
            continue

        if not content or len(content.strip()) < MIN_CHUNK_SIZE:
            if not dry_run:
                _register_file(collection, source_file, wing, agent)
            continue

        # Chunk — either exchange pairs or general extraction
        if extract_mode == "general":
            from .general_extractor import extract_memories

            chunks = extract_memories(content)
            # Each chunk already has memory_type; use it as the room name
        else:
            chunks = chunk_exchanges(content)

        if not chunks:
            if not dry_run:
                _register_file(collection, source_file, wing, agent)
            continue

        # Detect room from content (general mode uses memory_type instead)
        if extract_mode != "general":
            room = detect_convo_room(content)
        else:
            room = None  # set per-chunk below

        if dry_run:
            _report_dry_run_chunks(filepath, chunks, room, extract_mode, room_counts)
            total_drawers += len(chunks)
            continue

        if extract_mode != "general":
            room_counts[room] += 1

        # Lock + purge stale + file fresh chunks. Lock serializes concurrent
        # agents; purge removes pre-v2 drawers so the schema bump applies.
        drawers_added, room_delta, skipped = _file_chunks_locked(
            collection,
            source_file,
            chunks,
            wing,
            room,
            agent,
            extract_mode,
            closets_col=closets_col,
            smart_ctx=smart_ctx,
        )
        if skipped:
            files_skipped += 1
            continue
        for r, n in room_delta.items():
            room_counts[r] += n

        total_drawers += drawers_added
        print(f"  + [{i:4}/{len(files)}] {filepath.name[:50]:50} +{drawers_added}")

        # If cursor mode is enabled and this is a cursor-eligible source,
        # initialize the byte_cursor to the file's current safe boundary
        # so subsequent mines can resume incrementally instead of
        # re-processing the whole file.
        if cursor:
            _init_cursor_after_full_mine(collection, filepath, source_file, wing, agent)

    print(f"\n{'=' * 55}")
    print("  Done.")
    print(f"  Files processed: {len(files) - files_skipped}")
    print(f"  Files skipped (already filed): {files_skipped}")
    print(f"  Drawers filed: {total_drawers}")
    if room_counts:
        print("\n  By room:")
        for room, count in sorted(room_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"    {room:20} {count} files")
    print('\n  Next: mempalace search "what you\'re looking for"')
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convo_miner.py <convo_dir> [--palace PATH] [--limit N] [--dry-run]")
        sys.exit(1)
    from .config import MempalaceConfig

    mine_convos(sys.argv[1], palace_path=MempalaceConfig().palace_path)
