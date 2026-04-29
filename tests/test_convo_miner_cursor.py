"""Tests for cursor-based incremental ingest in convo_miner.

These tests describe the desired behavior of an append-only ingest path
for Claude Code and Codex CLI JSONL transcripts. They are intentionally
written before the implementation lands so they fail against current
main, then pass once the feature is implemented.

Design constraints the tests pin down:
  * Drawer IDs are content-addressed so re-mining is idempotent.
  * Cursor is stored on the existing _register_file sentinel record.
  * Cursor never advances past a "safe boundary" — defined by the
    normalize.py merge semantics for each format.
  * On first mine (cursor==0), the cursor path defers to the existing
    full-mine path. Cursor mode only handles incremental appends.
  * Any unexpected condition (unknown format, stale cursor, write
    failure) returns a "fallback" status so the existing full-mine
    path runs. Cursor mode must never prevent content from being
    mined.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

# Imports are deferred to inside each test so that import failure of any
# not-yet-implemented helper produces a clear test failure rather than a
# collection error that hides the rest of the suite.


# ────────────────────────────────────────────────────────────────────
# Synthetic fixtures
# ────────────────────────────────────────────────────────────────────


def _claude_user_line(text: str) -> str:
    """A Claude Code JSONL user-text line (string content, not tool_result)."""
    return json.dumps({"type": "user", "message": {"role": "user", "content": text}})


def _claude_assistant_line(text: str, tool_use=None) -> str:
    """A Claude Code JSONL assistant line, optionally with a tool_use block."""
    blocks = [{"type": "text", "text": text}]
    if tool_use:
        blocks.append(tool_use)
    return json.dumps({"type": "assistant", "message": {"role": "assistant", "content": blocks}})


def _claude_tool_result_line(tool_use_id: str, result_text: str) -> str:
    """A Claude Code JSONL user line containing only a tool_result block."""
    return json.dumps(
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": result_text,
                    }
                ],
            },
        }
    )


def _codex_session_meta() -> str:
    return json.dumps({"type": "session_meta", "id": "test-session"})


def _codex_event(role: str, text: str) -> str:
    payload_type = "user_message" if role == "user" else "agent_message"
    return json.dumps({"type": "event_msg", "payload": {"type": payload_type, "message": text}})


def _write_jsonl(path: Path, lines: list) -> None:
    """Write lines to a JSONL file with trailing newlines (the format Claude Code uses)."""
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, lines: list) -> None:
    """Append lines to an existing JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ────────────────────────────────────────────────────────────────────
# _is_cursor_eligible — format detection
# ────────────────────────────────────────────────────────────────────


class TestIsCursorEligible:
    def test_claude_code_jsonl_eligible(self, tmp_path):
        from mempalace.convo_miner import _is_cursor_eligible

        f = tmp_path / "session.jsonl"
        _write_jsonl(
            f,
            [
                _claude_user_line("hello"),
                _claude_assistant_line("hi back"),
            ],
        )
        assert _is_cursor_eligible(f) is True

    def test_codex_jsonl_eligible(self, tmp_path):
        from mempalace.convo_miner import _is_cursor_eligible

        f = tmp_path / "rollout.jsonl"
        _write_jsonl(f, [_codex_session_meta(), _codex_event("user", "hi")])
        assert _is_cursor_eligible(f) is True

    def test_arbitrary_jsonl_not_eligible(self, tmp_path):
        from mempalace.convo_miner import _is_cursor_eligible

        f = tmp_path / "data.jsonl"
        _write_jsonl(f, [json.dumps({"id": 1, "value": "x"})])
        assert _is_cursor_eligible(f) is False

    def test_markdown_not_eligible(self, tmp_path):
        from mempalace.convo_miner import _is_cursor_eligible

        f = tmp_path / "notes.md"
        f.write_text("# title\n\n> hi\nresponse", encoding="utf-8")
        assert _is_cursor_eligible(f) is False

    def test_empty_file_not_eligible(self, tmp_path):
        from mempalace.convo_miner import _is_cursor_eligible

        f = tmp_path / "empty.jsonl"
        f.write_text("", encoding="utf-8")
        assert _is_cursor_eligible(f) is False

    def test_missing_file_not_eligible(self, tmp_path):
        from mempalace.convo_miner import _is_cursor_eligible

        assert _is_cursor_eligible(tmp_path / "nope.jsonl") is False


# ────────────────────────────────────────────────────────────────────
# _find_safe_boundary — append-stable byte offset
# ────────────────────────────────────────────────────────────────────


class TestFindSafeBoundary:
    def test_advances_past_user_text_line(self, tmp_path):
        from mempalace.convo_miner import _find_safe_boundary

        # One user prompt line then an assistant turn. Boundary must be at
        # least past the user line, since user-text lines never receive
        # merges from later content.
        f = tmp_path / "s.jsonl"
        u = _claude_user_line("first prompt")
        a = _claude_assistant_line("response")
        _write_jsonl(f, [u, a])
        boundary = _find_safe_boundary(f)
        assert boundary >= len(u) + 1  # +1 for the newline

    def test_does_not_advance_past_tool_result_only_user(self, tmp_path):
        from mempalace.convo_miner import _find_safe_boundary

        # User line is tool_results-only, which the normalizer merges into
        # the previous assistant message. Cursor must NOT advance past it,
        # otherwise future appended content could change the merged message.
        f = tmp_path / "s.jsonl"
        u1 = _claude_user_line("first prompt")
        a1 = _claude_assistant_line(
            "running tool", tool_use={"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}
        )
        tr = _claude_tool_result_line("t1", "output")
        _write_jsonl(f, [u1, a1, tr])
        boundary = _find_safe_boundary(f)
        # Boundary must land on the u1 line, not after the tr line.
        assert boundary == len(u1) + 1

    def test_advances_through_multiple_user_turns(self, tmp_path):
        from mempalace.convo_miner import _find_safe_boundary

        f = tmp_path / "s.jsonl"
        lines = [
            _claude_user_line("turn1"),
            _claude_assistant_line("response1"),
            _claude_user_line("turn2"),
            _claude_assistant_line("response2"),
        ]
        _write_jsonl(f, lines)
        boundary = _find_safe_boundary(f)
        # Should advance past turn2 (the last user-text line).
        size = os.path.getsize(f)
        # Boundary must be past offset of turn2's start; safest assertion:
        # boundary >= (offset of turn2 line) + len(turn2) + 1
        prefix = "\n".join(lines[:2]) + "\n"
        expected_min = len(prefix.encode("utf-8")) + len(lines[2].encode("utf-8")) + 1
        assert boundary >= expected_min
        assert boundary <= size

    def test_codex_advances_per_event(self, tmp_path):
        from mempalace.convo_miner import _find_safe_boundary

        f = tmp_path / "rollout.jsonl"
        lines = [
            _codex_session_meta(),
            _codex_event("user", "q1"),
            _codex_event("assistant", "a1"),
            _codex_event("user", "q2"),
        ]
        _write_jsonl(f, lines)
        # Codex has no merging, so boundary should reach EOF (all lines complete).
        assert _find_safe_boundary(f) == os.path.getsize(f)

    def test_does_not_advance_into_partial_trailing_line(self, tmp_path):
        from mempalace.convo_miner import _find_safe_boundary

        # Write a complete line + a partial second line (no trailing newline).
        f = tmp_path / "s.jsonl"
        complete = _claude_user_line("done")
        partial = '{"type": "assistant", "message": {"role"'  # no closing newline, no closing brace
        f.write_bytes(complete.encode("utf-8") + b"\n" + partial.encode("utf-8"))
        boundary = _find_safe_boundary(f)
        # Boundary must not include any byte from the partial line.
        assert boundary <= len(complete) + 1

    def test_returns_zero_for_unrecognized_format(self, tmp_path):
        from mempalace.convo_miner import _find_safe_boundary

        f = tmp_path / "data.jsonl"
        _write_jsonl(f, [json.dumps({"foo": "bar"})])
        assert _find_safe_boundary(f) == 0


# ────────────────────────────────────────────────────────────────────
# Cursor read/write round-trip
# ────────────────────────────────────────────────────────────────────


class TestCursorIO:
    def test_read_returns_zero_when_sentinel_absent(self, palace_path):
        from mempalace.convo_miner import _read_cursor
        from mempalace.palace import get_collection

        col = get_collection(palace_path)
        assert _read_cursor(col, "/nope/file.jsonl") == 0

    def test_write_then_read_roundtrip(self, palace_path):
        from mempalace.convo_miner import _read_cursor, _write_cursor
        from mempalace.palace import get_collection

        col = get_collection(palace_path)
        _write_cursor(col, "/tmp/test.jsonl", "wing_test", "agent", 4096)
        assert _read_cursor(col, "/tmp/test.jsonl") == 4096

    def test_write_overwrites_prior_cursor(self, palace_path):
        from mempalace.convo_miner import _read_cursor, _write_cursor
        from mempalace.palace import get_collection

        col = get_collection(palace_path)
        _write_cursor(col, "/tmp/test.jsonl", "w", "a", 100)
        _write_cursor(col, "/tmp/test.jsonl", "w", "a", 999)
        assert _read_cursor(col, "/tmp/test.jsonl") == 999

    def test_read_handles_malformed_metadata(self, palace_path):
        """If the sentinel exists but byte_cursor is missing/malformed, return 0."""
        from mempalace.convo_miner import _read_cursor, _register_file
        from mempalace.palace import get_collection

        col = get_collection(palace_path)
        _register_file(col, "/tmp/t.jsonl", "wing_test", "agent")
        # Sentinel exists from _register_file but has no byte_cursor field.
        assert _read_cursor(col, "/tmp/t.jsonl") == 0


# ────────────────────────────────────────────────────────────────────
# _mine_with_cursor — control flow contracts
# ────────────────────────────────────────────────────────────────────


class TestMineWithCursorControlFlow:
    def test_first_mine_returns_fallback(self, palace_path, tmp_path):
        """When cursor==0 (no prior sentinel), return 'fallback' so the full
        mine path handles the initial pass."""
        from mempalace.convo_miner import _mine_with_cursor
        from mempalace.palace import get_collection

        f = tmp_path / "s.jsonl"
        _write_jsonl(
            f,
            [
                _claude_user_line("hi"),
                _claude_assistant_line("hello"),
                _claude_user_line("more"),
                _claude_assistant_line("ok"),
            ],
        )
        col = get_collection(palace_path)
        _drawers, _delta, status = _mine_with_cursor(col, str(f), "wing_t", "me", "exchange")
        assert status == "fallback"

    def test_cursor_at_or_past_safe_offset_skips(self, palace_path, tmp_path):
        from mempalace.convo_miner import _find_safe_boundary, _mine_with_cursor, _write_cursor
        from mempalace.palace import get_collection

        f = tmp_path / "s.jsonl"
        _write_jsonl(
            f,
            [
                _claude_user_line("a"),
                _claude_assistant_line("b"),
                _claude_user_line("c"),
                _claude_assistant_line("d"),
            ],
        )
        col = get_collection(palace_path)
        # Pre-write the cursor at the current safe boundary.
        sb = _find_safe_boundary(f)
        assert sb > 0
        _write_cursor(col, str(f), "w", "a", sb)
        _drawers, _delta, status = _mine_with_cursor(col, str(f), "w", "a", "exchange")
        assert status == "skipped"

    def test_unknown_format_returns_fallback(self, palace_path, tmp_path):
        from mempalace.convo_miner import _mine_with_cursor
        from mempalace.palace import get_collection

        f = tmp_path / "data.jsonl"
        _write_jsonl(f, [json.dumps({"foo": "bar"})])
        col = get_collection(palace_path)
        _drawers, _delta, status = _mine_with_cursor(col, str(f), "w", "a", "exchange")
        assert status == "fallback"

    def test_cursor_past_file_size_returns_fallback(self, palace_path, tmp_path):
        """File shrinkage / corruption: cursor exceeds file size → fallback."""
        from mempalace.convo_miner import _mine_with_cursor, _write_cursor
        from mempalace.palace import get_collection

        f = tmp_path / "s.jsonl"
        _write_jsonl(
            f,
            [_claude_user_line("hi"), _claude_assistant_line("ok")],
        )
        col = get_collection(palace_path)
        # Cursor far past the file's current size.
        _write_cursor(col, str(f), "w", "a", 99_999_999)
        _drawers, _delta, status = _mine_with_cursor(col, str(f), "w", "a", "exchange")
        assert status == "fallback"


# ────────────────────────────────────────────────────────────────────
# Drawer ID stability — content-addressed
# ────────────────────────────────────────────────────────────────────


class TestDrawerIdContentAddressed:
    def test_same_content_same_id_across_remines(self, palace_path, tmp_path):
        """The killer property: re-mining a file produces the same drawer
        IDs for unchanged content. Without this, cursor mode can't be
        idempotent."""
        from mempalace.convo_miner import mine_convos
        from mempalace.palace import get_collection

        # A small Claude Code transcript that produces real chunks.
        f = tmp_path / "s.jsonl"
        _write_jsonl(
            f,
            [
                _claude_user_line("explain auth flow please"),
                _claude_assistant_line("we use jwt tokens with refresh in cookies"),
                _claude_user_line("what about session storage"),
                _claude_assistant_line("redis with 24 hour ttl"),
            ],
        )
        wing = "wing_t"
        mine_convos(
            convo_dir=str(tmp_path),
            palace_path=palace_path,
            wing=wing,
            extract_mode="exchange",
            cursor=False,
        )
        col = get_collection(palace_path)
        first_ids = sorted(col.get(where={"source_file": str(f)}).get("ids", []))
        assert first_ids, "first mine should produce drawers"

        # Bump file mtime and re-mine via the schema rebuild path. With
        # content-addressed IDs, the same content must produce the same IDs.
        os.utime(f, None)
        # Force re-mine by deleting the version sentinel — simulates a schema bump.
        col.delete(where={"source_file": str(f)})
        mine_convos(
            convo_dir=str(tmp_path),
            palace_path=palace_path,
            wing=wing,
            extract_mode="exchange",
            cursor=False,
        )
        second_ids = sorted(col.get(where={"source_file": str(f)}).get("ids", []))
        assert second_ids == first_ids, (
            "drawer IDs must be content-addressed; unchanged content "
            "must produce identical IDs across mines"
        )

    def test_different_chunk_content_different_id(self, palace_path, tmp_path):
        from mempalace.convo_miner import mine_convos
        from mempalace.palace import get_collection

        f1 = tmp_path / "a.jsonl"
        f2 = tmp_path / "b.jsonl"
        _write_jsonl(f1, [_claude_user_line("alpha topic"), _claude_assistant_line("alpha reply")])
        _write_jsonl(f2, [_claude_user_line("beta topic"), _claude_assistant_line("beta reply")])
        mine_convos(convo_dir=str(tmp_path), palace_path=palace_path, wing="w", cursor=False)
        col = get_collection(palace_path)
        ids_a = set(col.get(where={"source_file": str(f1)}).get("ids", []))
        ids_b = set(col.get(where={"source_file": str(f2)}).get("ids", []))
        assert ids_a and ids_b
        assert ids_a.isdisjoint(ids_b), "different content must produce disjoint drawer IDs"


# ────────────────────────────────────────────────────────────────────
# End-to-end: mine_convos(cursor=True) with file growth
# ────────────────────────────────────────────────────────────────────


class TestMineConvosCursorEndToEnd:
    def test_first_mine_writes_cursor_after_full_mine(self, palace_path, tmp_path):
        from mempalace.convo_miner import _read_cursor, mine_convos
        from mempalace.palace import get_collection

        # Content has to clear MIN_CHUNK_SIZE (30 chars per exchange) for
        # the chunker to actually emit drawers — otherwise the full-mine
        # path register-files and continues without reaching the cursor
        # init step. Realistic for cursor mode anyway: short transcripts
        # don't benefit from incremental ingest.
        f = tmp_path / "s.jsonl"
        _write_jsonl(
            f,
            [
                _claude_user_line("explain the auth flow architecture please"),
                _claude_assistant_line("we use jwt tokens with refresh stored in httponly cookies"),
                _claude_user_line("what is the session storage backend"),
                _claude_assistant_line("redis with a 24 hour ttl per session record"),
            ],
        )
        mine_convos(convo_dir=str(tmp_path), palace_path=palace_path, wing="w", cursor=True)
        col = get_collection(palace_path)
        cursor = _read_cursor(col, str(f))
        assert cursor > 0, "cursor must be initialized after the first cursor-mode mine"

    def test_second_mine_unchanged_file_no_new_drawers(self, palace_path, tmp_path):
        from mempalace.convo_miner import mine_convos
        from mempalace.palace import get_collection

        f = tmp_path / "s.jsonl"
        _write_jsonl(
            f,
            [
                _claude_user_line("hello"),
                _claude_assistant_line("hi"),
                _claude_user_line("more"),
                _claude_assistant_line("ok"),
            ],
        )
        mine_convos(convo_dir=str(tmp_path), palace_path=palace_path, wing="w", cursor=True)
        col = get_collection(palace_path)
        ids_before = set(col.get(where={"source_file": str(f)}).get("ids", []))

        mine_convos(convo_dir=str(tmp_path), palace_path=palace_path, wing="w", cursor=True)
        ids_after = set(col.get(where={"source_file": str(f)}).get("ids", []))
        assert ids_after == ids_before, (
            "no new drawers should be added on re-mine of unchanged file"
        )

    def test_appending_lines_only_mines_new_content(self, palace_path, tmp_path):
        """The bug we're solving: append new exchanges to a transcript and
        re-mine. The original drawers' IDs must be preserved (proving they
        weren't re-embedded), and only new drawers should be added."""
        from mempalace.convo_miner import mine_convos
        from mempalace.palace import get_collection

        f = tmp_path / "s.jsonl"
        # Initial transcript with 2 user turns.
        _write_jsonl(
            f,
            [
                _claude_user_line("explain caching strategy in detail"),
                _claude_assistant_line(
                    "we use redis with a 24 hour ttl for session data and "
                    "memcached for query results"
                ),
                _claude_user_line("what about cache invalidation"),
                _claude_assistant_line(
                    "write-through with versioned keys, no manual invalidation needed"
                ),
            ],
        )
        mine_convos(convo_dir=str(tmp_path), palace_path=palace_path, wing="w", cursor=True)
        col = get_collection(palace_path)
        ids_initial = set(col.get(where={"source_file": str(f)}).get("ids", []))
        assert ids_initial, "initial mine must produce drawers"

        # Append two more user turns.
        _append_jsonl(
            f,
            [
                _claude_user_line("how do you handle stampedes"),
                _claude_assistant_line(
                    "request coalescing at the application layer, single flight pattern"
                ),
                _claude_user_line("any monitoring on cache hit rate"),
                _claude_assistant_line("prometheus metrics scraped every 15 seconds"),
            ],
        )
        mine_convos(convo_dir=str(tmp_path), palace_path=palace_path, wing="w", cursor=True)
        ids_after = set(col.get(where={"source_file": str(f)}).get("ids", []))

        # All original drawer IDs must still be present (not re-embedded).
        assert ids_initial.issubset(ids_after), (
            "original drawer IDs must survive the incremental mine — "
            "if they changed, the system re-embedded already-mined content"
        )
        # New content must produce new drawer(s).
        new_ids = ids_after - ids_initial
        assert new_ids, "appending new exchanges must produce new drawers"

    def test_cursor_mode_falls_back_for_markdown(self, palace_path, tmp_path):
        """A non-eligible source (markdown file with > markers) must still
        be mined correctly under cursor=True via the full-mine fallback."""
        from mempalace.convo_miner import _read_cursor, mine_convos
        from mempalace.palace import get_collection

        f = tmp_path / "notes.md"
        f.write_text(
            "> what is the architecture\n"
            "three-tier with postgres and redis\n\n"
            "> what about the deploy story\n"
            "github actions to ecs fargate\n\n"
            "> any caching\n"
            "redis with 24h ttl\n",
            encoding="utf-8",
        )
        mine_convos(convo_dir=str(tmp_path), palace_path=palace_path, wing="w", cursor=True)
        col = get_collection(palace_path)
        # Drawers must have been produced via the fallback full mine.
        ids = col.get(where={"source_file": str(f)}).get("ids", [])
        assert ids, "non-eligible source must still be mined via the fallback path"
        # No cursor should be written for an ineligible source.
        assert _read_cursor(col, str(f)) == 0

    def test_cursor_not_advanced_on_chunk_write_failure(self, palace_path, tmp_path):
        """If a drawer write fails mid-cursor, the cursor must NOT be
        advanced — next run reprocesses the same range, content-hash IDs
        make that idempotent."""
        from mempalace.convo_miner import _mine_with_cursor, _read_cursor, _write_cursor
        from mempalace.palace import get_collection

        f = tmp_path / "s.jsonl"
        # Initial content long enough to clear MIN_CHUNK_SIZE.
        initial = [
            _claude_user_line("describe the database migration approach in detail"),
            _claude_assistant_line(
                "alembic with autogenerate, reviewed by hand before each release"
            ),
        ]
        _write_jsonl(f, initial)
        col = get_collection(palace_path)
        # Position cursor at byte 1 so any new content is "tail" (the
        # tail will reparse some of the initial line, but the chunker
        # produces real drawers from the appended content).
        _write_cursor(col, str(f), "w", "a", 1)
        # Append new content so the cursor < safe_boundary precondition
        # holds AND the tail produces a chunkable transcript.
        _append_jsonl(
            f,
            [
                _claude_user_line("how do you handle long-running migrations safely"),
                _claude_assistant_line(
                    "online ddl with pt-online-schema-change, batched in 10k row chunks"
                ),
            ],
        )
        prior_cursor = _read_cursor(col, str(f))

        # Force a hard write failure inside the upsert path. The cursor must
        # not advance past the failing write.
        original_upsert = col._collection.upsert if hasattr(col, "_collection") else col.upsert

        def boom(*a, **kw):
            # Only fail on actual drawer upserts (not the sentinel cursor write).
            ids = kw.get("ids") or (a[1] if len(a) > 1 else [])
            if ids and any(str(i).startswith("drawer_") for i in ids):
                raise RuntimeError("simulated write failure")
            return original_upsert(*a, **kw)

        with patch.object(col, "upsert", side_effect=boom):
            _mine_with_cursor(col, str(f), "w", "a", "exchange")

        # The cursor must not have advanced past the prior position.
        assert _read_cursor(col, str(f)) == prior_cursor, (
            "cursor must not advance when drawer writes fail; otherwise "
            "the failed range is permanently skipped"
        )


# ────────────────────────────────────────────────────────────────────
# Real-transcript sanity check
# ────────────────────────────────────────────────────────────────────


@pytest.fixture
def real_claude_transcript(tmp_path):
    """Copy a small real Claude Code JSONL into tmp_path, or skip if none.

    conftest.py redirects HOME to an isolated temp dir, so Path.home()
    can't find real transcripts. We pull the pre-isolation HOME out of
    conftest._original_env. Skips cleanly if the user has no
    ~/.claude/projects directory or no small (<200 KB) transcript.

    This is a sanity check that the code handles real-world JSONL shape,
    not a behavioral test — assertion is just "it runs without error."
    """
    # conftest puts itself on sys.path via the rootdir; import directly.
    import conftest  # type: ignore[import-not-found]

    real_home = conftest._original_env.get("HOME")
    if not real_home:
        pytest.skip("no original HOME captured")
    base = Path(real_home) / ".claude" / "projects"
    if not base.is_dir():
        pytest.skip("no ~/.claude/projects directory")
    candidates = [p for p in base.rglob("*.jsonl") if 0 < p.stat().st_size < 200 * 1024]
    if not candidates:
        pytest.skip("no small real Claude Code transcript available")
    src = candidates[0]
    dst = tmp_path / "real.jsonl"
    dst.write_bytes(src.read_bytes())
    return dst


class TestRealTranscriptSanity:
    def test_real_transcript_mines_without_error(
        self, palace_path, tmp_path, real_claude_transcript
    ):
        """Mine a real Claude Code transcript with cursor=True and verify the
        run completes without exception. Doesn't assert specific drawer counts —
        real transcripts are noisy. The point is shape compatibility."""
        from mempalace.convo_miner import mine_convos

        # real_claude_transcript was placed in tmp_path; pass tmp_path to mine.
        mine_convos(convo_dir=str(tmp_path), palace_path=palace_path, wing="w_real", cursor=True)
        # If we got here without raising, the format is handled.

    def test_real_transcript_safe_boundary_in_range(self, real_claude_transcript):
        from mempalace.convo_miner import _find_safe_boundary

        size = real_claude_transcript.stat().st_size
        boundary = _find_safe_boundary(real_claude_transcript)
        assert 0 <= boundary <= size
