#!/usr/bin/env python3
import argparse
import json
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from fnmatch import fnmatch
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "palace_sync.config.json"
EXTENSIONS = (".jsonl", ".txt", ".md", ".json")
PENDING = ".palace-sync-pending"
PAD = "#" * 63 + "\n"

MANIFEST_PROG = r"""
import fnmatch, json, os, sys
root = sys.argv[1]
skip = sys.argv[2].split('|') if len(sys.argv) > 2 and sys.argv[2] else []
EXT = ('.jsonl', '.txt', '.md', '.json')
out = {}
for base, dirs, files in os.walk(root):
    dirs[:] = [d for d in dirs if d != 'tool-results']
    rel = os.path.relpath(base, root).replace(os.sep, '/')
    if rel == '.':
        continue
    for f in sorted(files):
        if not f.endswith(EXT) or f.endswith('.meta.json') or f.startswith('._'):
            continue
        if any(fnmatch.fnmatch(f, g) for g in skip):
            continue
        try:
            out[rel + '/' + f] = os.path.getsize(os.path.join(base, f))
        except OSError:
            pass
print(json.dumps(out))
"""

PACK_PROG = r"""
import hashlib, json, os, sys, tarfile
root, jobs = sys.argv[1], json.load(open(sys.argv[2]))
sys.stdout.buffer.write(b"\n" * 64 + b"PALACE-SYNC\n")
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|", format=tarfile.PAX_FORMAT) as out:
    for rel, off, end in jobs:
        with open(os.path.join(root, *rel.split("/")), "rb") as f:
            ti = tarfile.TarInfo(rel)
            if off:
                f.seek(max(off - 4096, 0))
                pre = hashlib.sha256(f.read(off - max(off - 4096, 0))).hexdigest()
                ti.size, ti.pax_headers = end - off, {"off": str(off), "pre": pre}
            else:
                ti.size = os.fstat(f.fileno()).st_size
            out.addfile(ti, f)
"""

UNPACK_PROG = r"""
import hashlib, json, os, shutil, sys, tarfile
mirror, done, diverged = sys.argv[1], 0, []
while sys.stdin.buffer.readline() not in (b"PALACE-SYNC\n", b""):
    pass
with tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as t:
    for m in t:
        p, off = os.path.join(mirror, *m.name.split("/")), int(m.pax_headers.get("off", 0))
        if off:
            ok = os.path.isfile(p) and os.path.getsize(p) == off
            if ok:
                with open(p, "rb") as f:
                    f.seek(max(off - 4096, 0))
                    ok = hashlib.sha256(f.read(off - max(off - 4096, 0))).hexdigest() == m.pax_headers["pre"]
            if not ok:
                diverged.append(m.name)
                continue
            with open(p, "ab") as f:
                shutil.copyfileobj(t.extractfile(m), f, 1 << 20)
        else:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p + ".part", "wb") as f:
                shutil.copyfileobj(t.extractfile(m), f, 1 << 20)
            os.replace(p + ".part", p)
        done += 1
print(json.dumps({"done": done, "diverged": diverged}))
"""


def scan(root: Path, skip=()):
    out = {}
    if not root.exists():
        return out
    for f in sorted(root.rglob("*")):
        if not f.is_file() or f.parent == root:
            continue
        if "tool-results" in f.relative_to(root).parts:
            continue
        if f.suffix.lower() not in EXTENSIONS or f.name.endswith(".meta.json"):
            continue
        if f.name.startswith("._") or any(fnmatch(f.name, g) for g in skip):
            continue
        out[f.relative_to(root).as_posix()] = f.stat().st_size
    return out


def wing_for(project_dir: str) -> str:
    from mempalace.config import normalize_wing_name

    return normalize_wing_name(project_dir)


def run(cmd, **kw):
    kw["input"] = kw["input"].encode() if "input" in kw else None
    p = subprocess.run(cmd, capture_output=True, **kw)
    p.stdout, p.stderr = (s.decode("utf-8", "replace") for s in (p.stdout, p.stderr))
    if p.returncode:
        raise SystemExit(
            f"exit {p.returncode} from {cmd[0]}\n"
            f"--- stdout ---\n{p.stdout[-4000:]}\n"
            f"--- stderr ---\n{p.stderr[-4000:]}"
        )
    return p


class Peer:
    def __init__(self, cfg, remote):
        self.__dict__.update(cfg)
        self.remote = remote
        self.shell = cfg.get("shell", "bash -s")
        self.pending = shlex.quote(f"{self.mirror_dir}/{PENDING}")

    def sh(self, script):
        if self.remote:
            return run(["ssh", "-o", "BatchMode=yes", self.ssh, self.shell], input=PAD + script)
        return run(shlex.split(self.shell), input=script)

    def py(self, prog, *args):
        argv = " ".join(map(shlex.quote, (self.python, "-", *args)))
        return f"{argv} <<'PYEOF'\n{prog}\nPYEOF\n"

    def seed(self, projects):
        names = " ".join(map(shlex.quote, projects))
        return (
            f"mkdir -p {shlex.quote(self.mirror_dir)}\n"
            f"test -e {self.pending} || printf '%s\\n' {names} > {self.pending}\n"
        )

    def stage(self, tag, prog, *args, data=None):
        base = f"{self.tmp_dir}/palace-sync-{tag}"
        argv = [self.python, f"{base}.py", *args] + ([f"{base}.json"] if data is not None else [])
        files = (
            {f"{base}.py": prog, f"{base}.json": json.dumps(data)}
            if data is not None
            else {f"{base}.py": prog}
        )
        if not self.remote:
            for path, text in files.items():
                Path(path).write_text(text)
            return argv
        files[f"{base}.sh"] = "exec " + " ".join(map(shlex.quote, argv)) + "\n"
        self.sh(
            "".join(f"cat > {shlex.quote(k)} <<'PSEOF'\n{v}\nPSEOF\n" for k, v in files.items())
        )
        bash = self.shell[: -len(" -s")] if self.shell.endswith(" -s") else self.shell
        return ["ssh", "-o", "BatchMode=yes", self.ssh, f"{bash} {base}.sh"]

    def queue(self, projects):
        if not projects:
            return []
        out = self.sh(self.seed(projects) + f"sort -u {self.pending}")
        return [p for p in out.stdout.splitlines() if p.strip()]

    def mine(self, proj: str):
        pre = f"{self.pre_mine} && " if getattr(self, "pre_mine", "") else ""
        self.sh(
            f"set -e\n{pre}PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8 "
            f"{shlex.quote(self.mempalace)} mine {shlex.quote(f'{self.mirror_dir}/{proj}')} "
            f"--mode convos --cursor --wing {shlex.quote(wing_for(proj))}\n"
            f"{{ grep -vxF -- {shlex.quote(proj)} {self.pending} || true; }} > {self.pending}.tmp\n"
            f"mv {self.pending}.tmp {self.pending}\n"
        )

    def _remote_scan(self, root, skip=()):
        return json.loads(self.sh(self.py(MANIFEST_PROG, root, "|".join(skip))).stdout)

    def manifest(self):
        skip = getattr(self, "exclude", ())
        if self.remote:
            return self._remote_scan(self.projects_dir, skip)
        return scan(Path(self.projects_dir), skip)

    def mirror_manifest(self):
        if self.remote:
            return self._remote_scan(self.mirror_dir)
        return scan(Path(self.mirror_dir))


def delta(src: dict, dst: dict):
    held = {k for k, n in src.items() if k.endswith(".jsonl") and dst.get(k, -1) > n}
    return sorted(k for k, n in src.items() if dst.get(k, -1) != n and k not in held), sorted(held)


def source_pairs(cfg, selected="all"):
    pairs = []
    sides = [cfg["local"], cfg["remote"]]
    codex_keys = ("codex_projects_dir", "codex_mirror_dir")
    has_codex = any(side.get(key) for side in sides for key in codex_keys)
    if has_codex and not all(side.get(key) for side in sides for key in codex_keys):
        raise SystemExit("Both peers require codex_projects_dir and codex_mirror_dir")
    for side in sides:
        if not side.get("projects_dir") or not side.get("mirror_dir"):
            raise SystemExit(f"{side['name']}: mirror_dir and projects_dir are both required")
        if has_codex:
            paths = [
                side[key].replace("\\", "/").rstrip("/").lower()
                for key in ("projects_dir", "mirror_dir", *codex_keys)
            ]
            for i, path in enumerate(paths):
                for other in paths[i + 1 :]:
                    if (
                        path == other
                        or path.startswith(other + "/")
                        or other.startswith(path + "/")
                    ):
                        raise SystemExit(
                            f"{side['name']}: transcript sources and mirrors must not overlap"
                        )
    if selected in ("all", "claude"):
        pairs.append(("claude", Peer(sides[0], False), Peer(sides[1], True)))
    if selected in ("all", "codex") and has_codex:
        peers = []
        for side, remote in zip(sides, (False, True)):
            settings = dict(
                side, projects_dir=side["codex_projects_dir"], mirror_dir=side["codex_mirror_dir"]
            )
            peers.append(Peer(settings, remote))
        pairs.append(("codex", *peers))
    elif selected == "codex":
        raise SystemExit("Codex sync is not configured on this pair")
    elif selected == "all" and not has_codex:
        print("Codex sync is not configured; syncing Claude only.")
    return pairs


def sync(src: Peer, dst: Peer, label: str, args) -> int:
    def say(line):
        print(f"[{label}] {line}", flush=True)

    have, mine = src.manifest(), dst.mirror_manifest()
    todo, held = delta(have, mine)
    new = [f for f in todo if f not in mine]
    say(f"{src.name} -> {dst.name}: {len(have)} transcripts, {len(mine)} in mirror")
    say(f"delta: {len(todo)} to ship ({len(new)} new, {len(todo) - len(new)} grown)")
    if held:
        say(f"held: {len(held)} shrank on {src.name}; keeping the longer mirror copy")
    if args.verbose:
        for f in todo[:20]:
            say(f"  {'NEW ' if f in new else 'GROW'} {f}")
        for f in held:
            say(f"  HELD {f}")
        if len(todo) > 20:
            say(f"  ... {len(todo) - 20} more")
    if args.dry_run:
        return 0
    projects = sorted({k.split("/")[0] for k in (*mine, *todo)})
    if todo:
        jobs = [[f, mine.get(f, 0) if f.endswith(".jsonl") else 0, have[f]] for f in todo]
        dst.sh(
            dst.seed(projects)
            + f"printf '%s\\n' {' '.join(map(shlex.quote, sorted({f.split('/')[0] for f in todo})))}"
            + f" >> {dst.pending}\n"
        )
        r = stream(src, dst, jobs, label)
        if r["diverged"]:
            say(f"{len(r['diverged'])} mirror copies diverged from source; shipping them whole")
            stream(src, dst, [[f, 0, have[f]] for f in r["diverged"]], label)
        say(f"shipped {sum(e - o for _, o, e in jobs):,} bytes")
    queue = dst.queue(projects)
    say(f"mining {len(queue)} of {len(projects)} projects")
    for proj in queue:
        say(f"mining {proj} -> wing {wing_for(proj)}")
        dst.mine(proj)
    return len(todo)


def stream(src: Peer, dst: Peer, jobs: list, label: str) -> dict:
    tag = label.lower().replace(" ", "-")
    pack = src.stage(f"{tag}-pack", PACK_PROG, src.projects_dir, data=jobs)
    unpack = dst.stage(f"{tag}-unpack", UNPACK_PROG, dst.mirror_dir)
    p = subprocess.Popen(pack, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
    q = subprocess.run(unpack, stdin=p.stdout, capture_output=True)
    p.stdout.close()
    p.wait()
    out, err = (s.decode("utf-8", "replace") for s in (q.stdout, q.stderr))
    if p.returncode or q.returncode:
        raise SystemExit(
            f"stream failed: pack exit {p.returncode}, unpack exit {q.returncode}\n{err[-4000:]}"
        )
    r = json.loads(out.strip().splitlines()[-1])
    if r["done"] + len(r["diverged"]) != len(jobs):
        raise SystemExit(
            f"stream applied {r['done']} + {len(r['diverged'])} diverged of {len(jobs)} files"
        )
    return r


def main():
    ap = argparse.ArgumentParser(description="Sync two MemPalace machines over SSH.")
    ap.add_argument("--pull-only", action="store_true", help="remote -> here only")
    ap.add_argument("--push-only", action="store_true", help="here -> remote only")
    ap.add_argument("--dry-run", action="store_true", help="report the delta, ship nothing")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument(
        "--source",
        choices=("all", "claude", "codex"),
        default="all",
        help="transcript source to sync (default: all configured sources)",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        sys.exit(f"no config at {cfg_path}\nCopy palace_sync.config.example.json and fill it in.")
    pairs = source_pairs(json.loads(cfg_path.read_text()), args.source)
    lanes = [
        [(there, here, f"PULL {s}") for s, here, there in pairs] * (not args.push_only),
        [(here, there, f"PUSH {s}") for s, here, there in pairs] * (not args.pull_only),
    ]
    moved, failed = [], []

    def drive(lane):
        try:
            moved.extend(sync(src, dst, label, args) for src, dst, label in lane)
        except BaseException as e:
            failed.append(e)
            print(f"\n{e}", file=sys.stderr, flush=True)

    with ThreadPoolExecutor(2) as ex:
        list(ex.map(drive, lanes))
    print(
        f"\ndone: {sum(moved)} transcript(s) moved"
        + (f", {len(failed)} lane(s) failed" if failed else "")
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
