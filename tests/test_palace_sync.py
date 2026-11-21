"""Keep both harnesses isolated while preserving stable mirror paths."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

spec = importlib.util.spec_from_file_location(
    "palace_sync", Path(__file__).parents[1] / "scripts/palace_sync.py"
)
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


def config(codex=True):
    result = {}
    for key, name, peer in [("local", "pc", "mac"), ("remote", "mac", "pc")]:
        result[key] = dict(
            name=name,
            projects_dir=f"/{name}/.claude/projects",
            mirror_dir=f"/{name}/{peer}-transcripts",
            tmp_dir="/tmp",
        )
        if codex:
            result[key].update(
                codex_projects_dir=f"/{name}/.codex/mempalace-transcripts",
                codex_mirror_dir=f"/{name}/{peer}-codex-transcripts",
            )
    return result


def test_legacy_config_retains_exact_paths(capsys):
    pairs = sync.source_pairs(config(False))
    assert len(pairs) == 1
    assert pairs[0][1].mirror_dir == "/pc/mac-transcripts"
    assert "Claude only" in capsys.readouterr().out


def test_both_sources_keep_separate_mirrors_and_origin_wings():
    pairs = sync.source_pairs(config())
    assert [p[0] for p in pairs] == ["claude", "codex"]
    assert pairs[0][1].mirror_dir == "/pc/mac-transcripts"
    assert pairs[1][1].mirror_dir == "/pc/mac-codex-transcripts"
    assert pairs[1][2].projects_dir == "/mac/.codex/mempalace-transcripts"
    assert pairs[1][2].remote and not pairs[1][1].remote
    assert sync.wing_for("-Users-peterwang-code-golfcore") == sync.wing_for(
        "_users_peterwang_code_golfcore"
    )


def test_codex_only_selection():
    assert [p[0] for p in sync.source_pairs(config(), "codex")] == ["codex"]
    with pytest.raises(SystemExit, match="not configured"):
        sync.source_pairs(config(False), "codex")


def test_partial_config_fails_before_sync():
    cfg = config()
    del cfg["remote"]["codex_mirror_dir"]
    with pytest.raises(SystemExit, match="Both peers require"):
        sync.source_pairs(cfg)


@pytest.mark.parametrize("mirror", ["/pc/mac-transcripts", "/pc/mac-transcripts/codex"])
def test_overlapping_mirrors_are_rejected(mirror):
    cfg = config()
    cfg["local"]["codex_mirror_dir"] = mirror
    with pytest.raises(SystemExit, match="must not overlap"):
        sync.source_pairs(cfg)


def test_local_and_remote_manifest_include_both_transcript_shapes(tmp_path):
    included = [
        "-Users-person-project/session.jsonl",
        "-Users-person-project/subagents/agent.jsonl",
        "_users_person_project/thread-id/transcript.jsonl",
    ]
    excluded = [
        "-Users-person-project/tool-results/tool.txt",
        "-Users-person-project/._session.jsonl",
        "-Users-person-project/session.meta.json",
    ]
    for relative in included + excluded:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    local = sync.scan(tmp_path)
    remote = subprocess.run(
        [sys.executable, "-c", sync.MANIFEST_PROG, str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert local == json.loads(remote.stdout)
    assert set(local) == set(included)


def test_delta_holds_shrunk_transcripts_and_ships_shrunk_notes():
    assert sync.delta({"new": 5, "grown": 10}, {"grown": 8}) == (["grown", "new"], [])
    assert sync.delta({"same": 5}, {"same": 5}) == ([], [])
    assert sync.delta({"a.jsonl": 3, "b.jsonl": 9}, {"a.jsonl": 8}) == (["b.jsonl"], ["a.jsonl"])
    assert sync.delta({"memory/x.md": 3}, {"memory/x.md": 8}) == (["memory/x.md"], [])


def shell_peer(tmp_path, name):
    mirror, log, stub = tmp_path / name, tmp_path / f"{name}.log", tmp_path / "mempalace"
    stub.write_text(f'#!/bin/sh\necho "$2" >> {log.as_posix()}\n')
    stub.chmod(0o755)
    peer = sync.Peer(
        dict(
            name=name,
            projects_dir=(tmp_path / "src").as_posix(),
            mirror_dir=mirror.as_posix(),
            tmp_dir=tmp_path.as_posix(),
            python=sys.executable,
            mempalace=stub.as_posix(),
            shell='"C:/Program Files/Git/bin/bash.exe" -s' if os.name == "nt" else "bash -s",
        ),
        False,
    )
    return peer, mirror, log


def test_pending_queue_seeds_everything_then_only_what_is_left(tmp_path):
    peer, mirror, log = shell_peer(tmp_path, "pc")
    assert peer.queue(["-a", "-b"]) == ["-a", "-b"]
    peer.mine("-a")
    assert peer.queue(["-a", "-b"]) == ["-b"]
    peer.mine("-b")
    assert peer.queue(["-a", "-b"]) == []
    assert log.read_text().split() == [f"{mirror.as_posix()}/-a", f"{mirror.as_posix()}/-b"]


def test_stream_appends_tails_replaces_notes_and_reships_divergence(tmp_path):
    dst, mirror, _ = shell_peer(tmp_path, "pc")
    src, _, _ = shell_peer(tmp_path, "mac")
    root = tmp_path / "src" / "-proj"
    root.mkdir(parents=True)
    (mirror / "-proj").mkdir(parents=True)
    for name, there, here in [
        ("grow.jsonl", "a\nb\n", "a\n"),
        ("fork.jsonl", "x\ny\n", "z\n"),
        ("note.md", "short", "much longer"),
        ("new.jsonl", "n\n", None),
    ]:
        (root / name).write_text(there)
        if here is not None:
            (mirror / "-proj" / name).write_text(here)
    have, mine = sync.scan(tmp_path / "src"), sync.scan(mirror)
    todo, held = sync.delta(have, mine)
    jobs = [[f, mine.get(f, 0) if f.endswith(".jsonl") else 0, have[f]] for f in todo]
    r = sync.stream(src, dst, jobs, "PUSH claude")
    assert r == {"done": 3, "diverged": ["-proj/fork.jsonl"]}
    sync.stream(src, dst, [[f, 0, have[f]] for f in r["diverged"]], "PUSH claude")
    assert sync.scan(mirror) == have and held == []
    assert (mirror / "-proj" / "grow.jsonl").read_text() == "a\nb\n"
    assert (mirror / "-proj" / "fork.jsonl").read_text() == "x\ny\n"
    assert not list(mirror.rglob("*.part"))


def test_stream_rejects_a_short_pack(tmp_path):
    dst, _, _ = shell_peer(tmp_path, "pc")
    src, _, _ = shell_peer(tmp_path, "mac")
    (tmp_path / "src" / "-proj").mkdir(parents=True)
    with pytest.raises(SystemExit, match="pack exit"):
        sync.stream(src, dst, [["-proj/missing.jsonl", 0, 5]], "PUSH claude")


def test_failed_mine_stays_queued(tmp_path):
    peer, _, _ = shell_peer(tmp_path, "pc")
    peer.mempalace = "false"
    peer.queue(["-a"])
    with pytest.raises(SystemExit):
        peer.mine("-a")
    assert peer.queue(["-a"]) == ["-a"]


def mangle(data: bytes) -> bytes:
    return b"\x80v\n" * 9 + data[9:]


def test_mangled_script_head_cannot_drop_set_e(tmp_path, monkeypatch):
    peer, _, _ = shell_peer(tmp_path, "pc")
    peer.queue(["-a"])
    real = sync.run

    def ssh(cmd, **kw):
        return real(["bash", "-s"], input=mangle(kw["input"].encode()).decode("latin-1"))

    peer.remote, peer.ssh, peer.mempalace = True, "pc", "false"
    monkeypatch.setattr(sync, "run", ssh)
    with pytest.raises(SystemExit):
        peer.mine("-a")
    monkeypatch.setattr(sync, "run", real)
    peer.remote = False
    assert peer.queue(["-a"]) == ["-a"]


def test_unpack_skips_a_mangled_stream_preamble(tmp_path):
    src, _, _ = shell_peer(tmp_path, "mac")
    (tmp_path / "src" / "-p").mkdir(parents=True)
    (tmp_path / "src" / "-p" / "s.jsonl").write_text("x\n")
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps([["-p/s.jsonl", 0, 2]]))
    packed = subprocess.run(
        [sys.executable, "-c", sync.PACK_PROG, str(tmp_path / "src"), str(jobs)],
        capture_output=True,
        check=True,
    ).stdout
    out = subprocess.run(
        [sys.executable, "-c", sync.UNPACK_PROG, str(tmp_path / "m")],
        input=mangle(packed),
        capture_output=True,
        check=True,
    )
    assert json.loads(out.stdout) == {"done": 1, "diverged": []}
    assert (tmp_path / "m" / "-p" / "s.jsonl").read_text() == "x\n"
