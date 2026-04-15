"""Tests for cursor-aware auto-ingest in hooks_cli.

These tests describe the desired behavior of the Stop and PreCompact
hooks once they wire up cursor-based incremental mining of the active
session transcript. They fail against current main and pass once the
hook integration lands.

Design:
  * `_maybe_auto_ingest(transcript_path, sync)` accepts the active
    transcript path. When given a valid one, it spawns
    `mempalace mine <parent_dir> --mode convos --cursor` so only newly
    appended JSONL lines get processed.
  * The pre-existing `MEMPAL_DIR` env-var fallback is preserved for
    backwards compatibility — both sources are evaluated independently.
  * `sync=True` (precompact path) blocks via subprocess.run; `sync=False`
    (stop hook path) backgrounds via subprocess.Popen.
  * `hook_stop` and `hook_precompact` plumb the transcript_path through
    to `_maybe_auto_ingest`.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch


# ────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────


def _write_real_jsonl(path: Path, n_user: int = 20):
    """Write a Claude Code-shaped JSONL with enough user turns to trigger
    the SAVE_INTERVAL=15 threshold inside hook_stop."""
    lines = []
    for i in range(n_user):
        lines.append(json.dumps({"message": {"role": "user", "content": f"q{i}"}}))
        lines.append(json.dumps({"message": {"role": "assistant", "content": f"a{i}"}}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ────────────────────────────────────────────────────────────────────
# _maybe_auto_ingest — transcript path support
# ────────────────────────────────────────────────────────────────────


class TestMaybeAutoIngestTranscript:
    def test_valid_transcript_spawns_cursor_mine(self, tmp_path):
        """When a real .jsonl transcript is passed, the function spawns
        `python -m mempalace mine <parent> --mode convos --cursor`."""
        from mempalace.hooks_cli import _maybe_auto_ingest

        transcript = tmp_path / "session.jsonl"
        _write_real_jsonl(transcript)

        with patch("mempalace.hooks_cli.subprocess.Popen") as mock_popen:
            with patch.dict(os.environ, {}, clear=True):
                _maybe_auto_ingest(transcript_path=str(transcript), sync=False)

        assert mock_popen.called, "background subprocess.Popen must be called"
        # Inspect the spawned command.
        call_args = mock_popen.call_args
        cmd = call_args.args[0] if call_args.args else call_args.kwargs.get("args", [])
        assert "mempalace" in cmd
        assert "mine" in cmd
        assert "--mode" in cmd
        idx = cmd.index("--mode")
        assert cmd[idx + 1] == "convos"
        assert "--cursor" in cmd
        # The directory passed to mine must be the transcript's parent.
        assert str(tmp_path) in cmd

    def test_sync_mode_uses_run_not_popen(self, tmp_path):
        """sync=True (precompact) blocks via subprocess.run; Popen is not called."""
        from mempalace.hooks_cli import _maybe_auto_ingest

        transcript = tmp_path / "session.jsonl"
        _write_real_jsonl(transcript)

        with (
            patch("mempalace.hooks_cli.subprocess.run") as mock_run,
            patch("mempalace.hooks_cli.subprocess.Popen") as mock_popen,
            patch.dict(os.environ, {}, clear=True),
        ):
            _maybe_auto_ingest(transcript_path=str(transcript), sync=True)

        assert mock_run.called, "sync=True must use subprocess.run"
        assert not mock_popen.called, "sync=True must not background via Popen"

    def test_invalid_transcript_path_no_spawn(self, tmp_path):
        """A path that fails validation (wrong extension, traversal, missing)
        must not spawn anything when MEMPAL_DIR is also unset."""
        from mempalace.hooks_cli import _maybe_auto_ingest

        bad_path = tmp_path / "notes.txt"  # wrong extension
        bad_path.write_text("hi", encoding="utf-8")

        with (
            patch("mempalace.hooks_cli.subprocess.Popen") as mock_popen,
            patch("mempalace.hooks_cli.subprocess.run") as mock_run,
            patch.dict(os.environ, {}, clear=True),
        ):
            _maybe_auto_ingest(transcript_path=str(bad_path), sync=False)

        assert not mock_popen.called
        assert not mock_run.called

    def test_neither_transcript_nor_mempal_dir_no_spawn(self):
        """No transcript, no env var → nothing happens, no error."""
        from mempalace.hooks_cli import _maybe_auto_ingest

        with (
            patch("mempalace.hooks_cli.subprocess.Popen") as mock_popen,
            patch("mempalace.hooks_cli.subprocess.run") as mock_run,
            patch.dict(os.environ, {}, clear=True),
        ):
            _maybe_auto_ingest(transcript_path="", sync=False)

        assert not mock_popen.called
        assert not mock_run.called

    def test_both_transcript_and_mempal_dir_spawn_both(self, tmp_path):
        """When MEMPAL_DIR is also set, both ingest sources fire (existing
        behavior preserved + new transcript path)."""
        from mempalace.hooks_cli import _maybe_auto_ingest

        transcript = tmp_path / "session.jsonl"
        _write_real_jsonl(transcript)
        mempal_dir = tmp_path / "configured"
        mempal_dir.mkdir()

        with (
            patch("mempalace.hooks_cli.subprocess.Popen") as mock_popen,
            patch.dict(os.environ, {"MEMPAL_DIR": str(mempal_dir)}, clear=True),
        ):
            _maybe_auto_ingest(transcript_path=str(transcript), sync=False)

        assert mock_popen.call_count == 2, (
            "both the active transcript and the MEMPAL_DIR fallback must spawn "
            "ingests when both are configured"
        )
        # One call should target the transcript's parent with --cursor.
        # One call should target MEMPAL_DIR (without --cursor).
        cmds = [c.args[0] if c.args else c.kwargs["args"] for c in mock_popen.call_args_list]
        cursor_cmds = [c for c in cmds if "--cursor" in c]
        non_cursor_cmds = [c for c in cmds if "--cursor" not in c]
        assert len(cursor_cmds) == 1
        assert len(non_cursor_cmds) == 1
        assert str(mempal_dir) in non_cursor_cmds[0]

    def test_mempal_dir_only_preserves_existing_behavior(self, tmp_path):
        """No transcript, only MEMPAL_DIR → preserves the original
        `python -m mempalace mine <dir>` invocation (no cursor flag)."""
        from mempalace.hooks_cli import _maybe_auto_ingest

        mempal_dir = tmp_path / "configured"
        mempal_dir.mkdir()

        with (
            patch("mempalace.hooks_cli.subprocess.Popen") as mock_popen,
            patch.dict(os.environ, {"MEMPAL_DIR": str(mempal_dir)}, clear=True),
        ):
            _maybe_auto_ingest(transcript_path="", sync=False)

        assert mock_popen.called
        cmd = mock_popen.call_args.args[0]
        assert str(mempal_dir) in cmd
        assert "--cursor" not in cmd  # original behavior is full mine


# ────────────────────────────────────────────────────────────────────
# hook_stop & hook_precompact — plumbing
# ────────────────────────────────────────────────────────────────────


class TestHookPlumbing:
    def test_hook_stop_passes_transcript_path_to_auto_ingest(self, tmp_path):
        """hook_stop must call _maybe_auto_ingest with the transcript_path
        parsed from stdin so the cursor mine targets the active session."""
        from mempalace.hooks_cli import hook_stop

        transcript = tmp_path / "session.jsonl"
        _write_real_jsonl(transcript)  # writes 20 user msgs > SAVE_INTERVAL

        # Force the save threshold to fire — fresh session, no prior save.
        # Use a unique session_id so STATE_DIR/{sid}_last_save is absent.
        data = {
            "session_id": "test-session-cursor-plumbing",
            "stop_hook_active": False,
            "transcript_path": str(transcript),
        }

        with patch("mempalace.hooks_cli._maybe_auto_ingest") as mock_ai:
            # Suppress stdout from _output to keep the test quiet.
            with patch("builtins.print"):
                hook_stop(data, "claude-code")

        assert mock_ai.called, "_maybe_auto_ingest must be called when threshold fires"
        # Verify transcript_path was passed.
        call_kwargs = mock_ai.call_args.kwargs
        # Accept either positional or keyword for resilience to signature shape.
        passed = call_kwargs.get("transcript_path") or (
            mock_ai.call_args.args[0] if mock_ai.call_args.args else None
        )
        assert passed == str(transcript)

    def test_hook_precompact_passes_transcript_path_with_sync_true(self, tmp_path):
        """hook_precompact must call _maybe_auto_ingest with sync=True so
        memories land before context compaction destroys them."""
        from mempalace.hooks_cli import hook_precompact

        transcript = tmp_path / "session.jsonl"
        _write_real_jsonl(transcript, n_user=2)

        data = {
            "session_id": "test-session-precompact",
            "transcript_path": str(transcript),
        }

        with patch("mempalace.hooks_cli._maybe_auto_ingest") as mock_ai:
            with patch("builtins.print"):
                hook_precompact(data, "claude-code")

        assert mock_ai.called
        kwargs = mock_ai.call_args.kwargs
        # Validate transcript_path and sync=True were passed.
        passed_path = kwargs.get("transcript_path") or (
            mock_ai.call_args.args[0] if mock_ai.call_args.args else None
        )
        assert passed_path == str(transcript)
        assert kwargs.get("sync") is True


# ────────────────────────────────────────────────────────────────────
# Defensive properties
# ────────────────────────────────────────────────────────────────────


class TestAutoIngestDefensive:
    def test_subprocess_oserror_does_not_propagate(self, tmp_path):
        """If subprocess.Popen raises OSError (e.g., python not on PATH),
        the hook must not crash — it just silently skips ingest."""
        from mempalace.hooks_cli import _maybe_auto_ingest

        transcript = tmp_path / "session.jsonl"
        _write_real_jsonl(transcript)

        with (
            patch(
                "mempalace.hooks_cli.subprocess.Popen",
                side_effect=OSError("no python"),
            ),
            patch.dict(os.environ, {}, clear=True),
        ):
            # Must not raise.
            _maybe_auto_ingest(transcript_path=str(transcript), sync=False)
