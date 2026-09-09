"""Keep both harnesses isolated while preserving stable mirror paths."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

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


def test_delta_preserves_longer_mirror():
    assert sync.delta({"new": 5, "grown": 10}, {"grown": 8}) == ["grown", "new"]
    assert sync.delta({"same": 5}, {"same": 5}) == []
    with pytest.raises(SystemExit, match="preserving longer"):
        sync.delta({"truncated": 3}, {"truncated": 8})


def test_no_transfer_still_retries_unfinished_mine():
    manifest = {"_users_person_project/thread-id/transcript.jsonl": 100}
    commands = []
    src = SimpleNamespace(name="mac", manifest=lambda: manifest)
    dst = SimpleNamespace(
        name="pc",
        mirror_manifest=lambda: manifest,
        mirror_dir="/pc/mac-codex-transcripts",
        mempalace="mempalace",
        sh=commands.append,
    )
    args = SimpleNamespace(verbose=False, dry_run=False)
    assert sync.sync(src, dst, "PULL codex", args) == 0
    assert len(commands) == 1
    assert "/pc/mac-codex-transcripts/_users_person_project" in commands[0]
    assert "--cursor --wing _users_person_project" in commands[0]
