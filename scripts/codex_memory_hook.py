"""Codex Stop/SessionEnd bridge for personal-v3's incremental miner.

Reads Codex history databases read-only. Exports authored conversation text in
the miner's supported JSONL shape at a permanent per-project, per-thread path.
The original Codex item is retained alongside it. Never mine reasoning or
synthetic response_item messages. Hook entry detaches before doing database work.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time


def readonly(path):
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)


def authored(item):
    kind = item.get("type")
    if kind == "agentMessage":
        text = item.get("text", "")
        return "assistant", text if isinstance(text, str) else ""
    if kind == "userMessage":
        return "user", "\n".join(
            p["text"]
            for p in item.get("content", [])
            if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str)
        )
    return None, ""


def history(home, session_id, transcript=None):
    db = home / "thread_history_1.sqlite"
    if db.exists():
        with readonly(db) as connection:
            rows = connection.execute(
                "SELECT item_json, created_at_ms FROM thread_items "
                "WHERE thread_id=? AND item_type IN ('userMessage','agentMessage') "
                "ORDER BY rollout_ordinal",
                (session_id,),
            ).fetchall()
        if rows:
            return [(json.loads(raw), stamp) for raw, stamp in rows]
    result = []
    if transcript and Path(transcript).is_file():
        for line in Path(transcript).read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A running rollout may have a partially written final line.
            if not isinstance(row, dict):
                continue
            payload = row.get("payload", {})
            if row.get("type") != "event_msg" or not isinstance(payload, dict):
                continue
            if payload.get("type") == "item_completed":
                result.append((payload.get("item", {}), payload.get("completed_at_ms")))
            elif payload.get("type") in ("user_message", "agent_message"):
                text = payload.get("message", "")
                item = {"id": hashlib.sha256(line.encode()).hexdigest()}
                if payload["type"] == "user_message":
                    item.update(type="userMessage", content=[{"type": "text", "text": text}])
                else:
                    item.update(type="agentMessage", text=text)
                result.append((item, None))
    return result


def export(home, payload):
    from mempalace.config import normalize_wing_name

    session_id = payload["session_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", session_id):
        raise ValueError("Invalid Codex session id")
    cwd = payload.get("cwd")
    if not cwd:
        raise ValueError("Codex hook must provide its project cwd")
    # Match the existing machine-origin project wings, never the YYYY/MM/DD
    # rollout directory. This path is permanent: changing it duplicates IDs.
    wing = normalize_wing_name(re.sub(r"[:/\\]", "_", cwd))
    destination = home / "mempalace-transcripts" / wing / session_id / "transcript.jsonl"
    records = []
    for item, stamp in history(home, session_id, payload.get("transcript_path")):
        if not isinstance(item, dict):
            continue
        role, text = authored(item)
        if role and text:
            records.append(
                {
                    "type": role,
                    "uuid": item["id"],
                    "message": {"role": role, "content": text},
                    "codex_item": item,
                    "created_at_ms": stamp,
                }
            )
    previous = []
    if destination.exists():
        previous = [
            json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()
        ]
    if records[: len(previous)] != previous:
        raise RuntimeError(
            "Codex history changed before the saved cursor; preserving existing transcript"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as out:
        for record in records[len(previous) :]:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    return destination, wing, len(records), len(records) - len(previous)


@contextmanager
def worker_lock(home):
    # Serialize export + cursor mining across hook events; the miner separately
    # coordinates its palace writes with other Mempalace processes.
    path = home / "mempalace-hook.lock"
    with path.open("a+b") as handle:
        handle.seek(0)
        handle.write(b"0")
        handle.flush()
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    home.mkdir(parents=True, exist_ok=True)
    if args.worker:
        payload = json.loads(args.worker.read_text(encoding="utf-8"))
    else:
        payload = json.load(sys.stdin)
    if args.worker or args.export_only:
        with worker_lock(home):
            path, wing, count, added = export(home, payload)
            print(
                json.dumps({"path": str(path), "wing": wing, "messages": count, "added": added}),
                flush=True,
            )
            if count and not args.export_only:
                subprocess.run(
                    [
                        sys.executable,
                        "-X",
                        "utf8",
                        "-m",
                        "mempalace",
                        "mine",
                        str(path.parent),
                        "--mode",
                        "convos",
                        "--cursor",
                        "--wing",
                        wing,
                    ],
                    check=True,
                    timeout=600,
                )
        if args.worker:
            args.worker.unlink()
        return
    pending = home / "mempalace-hook-inputs"
    pending.mkdir(parents=True, exist_ok=True)
    request = pending / f"{time.time_ns()}-{os.getpid()}.json"
    request.write_text(json.dumps(payload), encoding="utf-8")
    options = (
        {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    with (home / "mempalace-hooks.log").open("a", encoding="utf-8") as log:
        subprocess.Popen(
            [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--worker", str(request)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            close_fds=True,
            **options,
        )
    print("{}")


if __name__ == "__main__":
    main()
