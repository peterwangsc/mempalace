"""Current Codex history, rollout, and append-only hook regression tests."""

import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest

from mempalace.normalize import _try_codex_jsonl
from mempalace.hooks_cli import _count_human_messages

spec = importlib.util.spec_from_file_location(
    "codex_memory_hook", Path(__file__).parents[1] / "scripts/codex_memory_hook.py"
)
hook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)


def user(identifier="u1", text="Keep this verbatim: 日本語"):
    return {"type": "userMessage", "id": identifier, "content": [{"type": "text", "text": text}]}


def agent():
    return {"type": "agentMessage", "id": "a1", "text": "Saved exactly.", "phase": "final_answer"}


def put(db, item, ordinal):
    db.execute(
        "INSERT INTO thread_items VALUES (?,?,?,?,?)",
        ("session-1", item["type"], json.dumps(item), ordinal, ordinal * 1000),
    )
    db.commit()


@pytest.fixture
def history_db(tmp_path):
    db = sqlite3.connect(tmp_path / "thread_history_1.sqlite")
    db.execute(
        "CREATE TABLE thread_items (thread_id, item_type, item_json, rollout_ordinal, created_at_ms)"
    )
    yield db
    db.close()


def test_history_export_is_incremental_and_read_only(tmp_path, history_db):
    put(history_db, user(), 1)
    put(history_db, {"type": "reasoning", "id": "r1", "text": "private"}, 2)
    put(history_db, agent(), 3)
    payload = {"session_id": "session-1", "cwd": "C:\\Users\\pewa\\code\\golfcore"}
    path, wing, count, added = hook.export(tmp_path, payload)
    assert (wing, count, added) == ("c__users_pewa_code_golfcore", 2, 2)
    original = path.read_bytes()
    assert "日本語" in original.decode()
    assert b"private" not in original
    assert hook.export(tmp_path, payload)[-1] == 0
    assert path.read_bytes() == original
    put(history_db, user("u2", "Next turn"), 4)
    assert hook.export(tmp_path, payload)[-1] == 1
    assert path.read_bytes().startswith(original)
    with hook.readonly(tmp_path / "thread_history_1.sqlite") as connection:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("DELETE FROM thread_items")


def test_changed_history_preserves_saved_text(tmp_path, history_db):
    put(history_db, user(), 1)
    payload = {"session_id": "session-1", "cwd": "/work/project"}
    path, *_ = hook.export(tmp_path, payload)
    original = path.read_bytes()
    history_db.execute("DELETE FROM thread_items")
    history_db.commit()
    put(history_db, user(text="Rewritten"), 1)
    with pytest.raises(RuntimeError, match="preserving"):
        hook.export(tmp_path, payload)
    assert path.read_bytes() == original


def test_current_rollout_parser_and_counter(tmp_path):
    rows = [{"type": "session_meta", "payload": {}}]
    for item in [user(), {"type": "reasoning", "id": "r1", "text": "private"}, agent()]:
        rows.append({"type": "event_msg", "payload": {"type": "item_completed", "item": item}})
    rows.append({"type": "response_item", "payload": {"type": "message", "text": "synthetic"}})
    content = "\n".join(map(json.dumps, rows))
    path = tmp_path / "rollout.jsonl"
    path.write_text(content + '\n{"partial":', encoding="utf-8")
    normalized = _try_codex_jsonl(content)
    assert "日本語" in normalized and "Saved exactly." in normalized
    assert "private" not in normalized and "synthetic" not in normalized
    assert _count_human_messages(str(path)) == 1
    authored = [hook.authored(item) for item, _ in hook.history(tmp_path, "session-1", path)]
    assert [text for role, text in authored if role] == [
        user()["content"][0]["text"],
        agent()["text"],
    ]


def test_legacy_rollout_fallback(tmp_path):
    path = tmp_path / "legacy.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"type": "event_msg", "payload": {"type": kind, "message": text}})
            for kind, text in [("user_message", "hello"), ("agent_message", "world")]
        ),
        encoding="utf-8",
    )
    assert [hook.authored(item) for item, _ in hook.history(tmp_path, "session-1", path)] == [
        ("user", "hello"),
        ("assistant", "world"),
    ]


def test_export_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError, match="session id"):
        hook.export(tmp_path, {"session_id": "../elsewhere", "cwd": "/work/project"})


def test_detached_entry_returns_without_mining(tmp_path, monkeypatch, capsys):
    import io

    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.setattr(hook.sys, "argv", ["hook"])
    monkeypatch.setattr(
        hook.sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {"session_id": "session-1", "cwd": "/work/project", "hook_event_name": "SessionEnd"}
            )
        ),
    )
    calls = []
    monkeypatch.setattr(
        hook.subprocess, "Popen", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    hook.main()
    assert capsys.readouterr().out.strip() == "{}"
    assert len(calls) == 1
    assert "--worker" in calls[0][0][0]
    assert calls[0][1]["stdin"] == hook.subprocess.DEVNULL
    assert len(list((tmp_path / "mempalace-hook-inputs").glob("*.json"))) == 1
