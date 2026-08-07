#!/usr/bin/env python3
"""Two-machine palace sync over SSH. Run from either side; it does both directions.

    python scripts/palace_sync.py                 # pull then push
    python scripts/palace_sync.py --pull-only
    python scripts/palace_sync.py --dry-run

Transcripts are the source of truth and the vector DB is derived, so this ships
*.jsonl only and re-mines. It never touches chroma.sqlite3 â€” there is no import
path and hand-editing has corrupted a palace before.

Every rule below is here because breaking it cost real damage once:

  jsonl only        `~/.claude/projects/<slug>/` also holds memory/*.md and
                    tool-results/*.txt, which mine as if they were conversations.
                    Tarring whole directories shipped 227 junk drawers on
                    2026-07-27 and 33 more on 2026-08-06.
  delta only        A full ship is 214 MB and minutes of CPU the cursor then
                    skips. Compare sizes against the mirror and send the rest.
  fixed mirrors     drawer_id = sha256(source_file + chunk)[:24] over the FULL
                    ABSOLUTE path, so a mirror that moves refiles the entire
                    corpus as duplicates. The script refuses to invent one.
  origin wing       The wing records which machine a conversation happened on.
                    Derived from the project directory, never guessed.
  ./ prefixes       Mac project dirs start with '-', which tar and find parse as
                    a flag and then silently match nothing.
  ulimit -n 10240   macOS defaults to 256; chroma reopens its client mid-mine and
                    dies with "Too many open files".
"""
import argparse, json, os, shlex, subprocess, sys
from fnmatch import fnmatch
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "palace_sync.config.json"

# Emitted on the far side to inventory transcripts. Sent over stdin so no
# quoting survives a shell, and written in py3.6-compatible syntax.
MANIFEST_PROG = r"""
import fnmatch, json, os, sys
root = sys.argv[1]
skip = sys.argv[2].split('|') if len(sys.argv) > 2 and sys.argv[2] else []
out = {}
# One level deep, deliberately. `<project>/<session-uuid>/tool-results/*.txt`
# lives below this and mines into a wing per session; walking recursively is
# how ~120 uuid wings got into the palace.
for d in sorted(os.listdir(root)):
    p = os.path.join(root, d)
    if not os.path.isdir(p):
        continue
    for f in sorted(os.listdir(p)):
        if not f.endswith('.jsonl') or f.startswith('._'):
            continue
        if any(fnmatch.fnmatch(f, g) for g in skip):
            continue
        try:
            out[d + '/' + f] = os.path.getsize(os.path.join(p, f))
        except OSError:
            pass
print(json.dumps(out))
"""

# Subagent transcripts. Their drawers are real content but they are dominated by
# tool traffic, and Peter's call on 2026-08-06 is to leave them out.
DEFAULT_EXCLUDE = ["agent-*.jsonl"]


def wing_for(project_dir: str) -> str:
    """`-Users-peterwang-code-golfcore` -> `_users_peterwang_code_golfcore`,
    `C--Users-pewa-code-golfcore` -> `c__users_pewa_code_golfcore`."""
    return project_dir.lower().replace("-", "_")


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)


class Peer:
    def __init__(self, cfg, remote):
        self.__dict__.update(cfg)
        self.remote = remote

    def sh(self, script, stdin=None):
        """Run a shell snippet here or over ssh."""
        if self.remote:
            return run(["ssh", "-o", "BatchMode=yes", self.ssh, script], input=stdin)
        shell = ["bash", "-lc", script] if os.name != "nt" else ["powershell", "-NoProfile", "-Command", script]
        return run(shell, input=stdin)

    def manifest(self):
        """{'<project>/<file>.jsonl': size} for every transcript worth mining."""
        skip = getattr(self, "exclude", None) or DEFAULT_EXCLUDE
        if self.remote:
            r = self.sh(f"{shlex.quote(self.python)} - {shlex.quote(self.projects_dir)} "
                        f"{shlex.quote('|'.join(skip))}", stdin=MANIFEST_PROG)
            return json.loads(r.stdout)
        out = {}
        root = Path(self.projects_dir)
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            for f in sorted(d.glob("*.jsonl")):
                if f.name.startswith("._") or any(fnmatch(f.name, g) for g in skip):
                    continue
                out[f"{d.name}/{f.name}"] = f.stat().st_size
        return out

    def mirror_manifest(self):
        """Same shape, for the copy of the *other* machine's transcripts held here."""
        if self.remote:
            r = self.sh(f"{shlex.quote(self.python)} - {shlex.quote(self.mirror_dir)}", stdin=MANIFEST_PROG)
            return json.loads(r.stdout)
        out = {}
        root = Path(self.mirror_dir)
        if not root.exists():
            return out
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            for f in sorted(d.glob("*.jsonl")):
                out[f"{d.name}/{f.name}"] = f.stat().st_size
        return out


def delta(src: dict, dst: dict):
    """Files the destination is missing or holds a shorter copy of.

    Transcripts are append-only, so a size match means nothing new. This is the
    materialised form of the cursor: the mirror is how far the far side has got.
    """
    return sorted(k for k, n in src.items() if dst.get(k, -1) != n)


def sync(src: Peer, dst: Peer, label: str, args) -> int:
    print(f"\n=== {label}: {src.name} -> {dst.name}")
    have, mine = src.manifest(), dst.mirror_manifest()
    todo = delta(have, mine)
    new = [f for f in todo if f not in mine]
    print(f"    {len(have)} transcripts on {src.name}, {len(mine)} in {dst.name}'s mirror")
    print(f"    delta: {len(todo)} to ship ({len(new)} new, {len(todo) - len(new)} grown)")
    if not todo:
        print("    nothing to do")
        return 0
    if args.verbose:
        for f in todo[:20]:
            print(f"      {'NEW ' if f in new else 'GROW'} {f}")
        if len(todo) > 20:
            print(f"      ... {len(todo) - 20} more")
    if args.dry_run:
        print("    (dry run, stopping here)")
        return 0

    stamp = os.environ.get("PALACE_SYNC_STAMP", "sync")
    tar_name = f"palace-sync-{src.name}-to-{dst.name}-{stamp}.tar.gz"
    src_tar = f"{src.tmp_dir}/{tar_name}"
    dst_tar = f"{dst.tmp_dir}/{tar_name}"

    # Build the archive from an explicit file list: nothing but the delta, and
    # nothing that is not a transcript, can end up inside it.
    listing = "\n".join("./" + f for f in todo) + "\n"
    src.sh(
        f"cd {shlex.quote(src.projects_dir)} && cat > {shlex.quote(src_tar)}.list && "
        f"tar czf {shlex.quote(src_tar)} -T {shlex.quote(src_tar)}.list && "
        f"rm -f {shlex.quote(src_tar)}.list"
        if src.remote or os.name != "nt"
        else f"Set-Location {src.projects_dir}; "
        f"$l=[Console]::In.ReadToEnd(); Set-Content -Encoding ascii '{src_tar}.list' $l; "
        f"tar czf '{src_tar}' -T '{src_tar}.list'; Remove-Item '{src_tar}.list'",
        stdin=listing,
    )

    # Move it. scp needs one local and one remote endpoint.
    if src.remote and not dst.remote:
        run(["scp", "-q", f"{src.ssh}:{src_tar}", dst_tar])
    elif dst.remote and not src.remote:
        run(["scp", "-q", src_tar, f"{dst.ssh}:{dst_tar}"])
    else:
        raise SystemExit("one side must be local")

    # Integrity, because a truncated archive extracts to a plausible-looking subset.
    a = src.sh(f"{shlex.quote(src.python)} -c \"import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())\" {shlex.quote(src_tar)}"
               if src.remote else
               f"$h=Get-FileHash '{src_tar}' -Algorithm SHA256; $h.Hash.ToLower()")
    b = dst.sh(f"{shlex.quote(dst.python)} -c \"import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())\" {shlex.quote(dst_tar)}"
               if dst.remote else
               f"$h=Get-FileHash '{dst_tar}' -Algorithm SHA256; $h.Hash.ToLower()")
    if a.stdout.strip().lower() != b.stdout.strip().lower():
        raise SystemExit(f"checksum mismatch shipping {tar_name} â€” aborting before extract")
    print(f"    shipped {tar_name}, sha256 verified")

    # Extract into the permanent mirror. Never anywhere else.
    if dst.remote:
        dst.sh(f"mkdir -p {shlex.quote(dst.mirror_dir)} && "
               f"tar xzf {shlex.quote(dst_tar)} -C {shlex.quote(dst.mirror_dir)} && "
               f"find {shlex.quote(dst.mirror_dir)} -name '._*' -delete && "
               f"rm -f {shlex.quote(dst_tar)}")
    else:
        dst.sh(f"New-Item -ItemType Directory -Force '{dst.mirror_dir}' | Out-Null; "
               f"tar xzf '{dst_tar}' -C '{dst.mirror_dir}'; "
               f"Get-ChildItem '{dst.mirror_dir}' -Recurse -Filter '._*' -File -EA SilentlyContinue | Remove-Item -Force; "
               f"Remove-Item '{dst_tar}' -Force")

    # `mempalace mine <dir>` walks the directory, so filtering the archive is not
    # enough on its own: anything already sitting in the mirror gets mined too.
    # `<project>/<uuid>/subagents/agent-*.jsonl` is the case that mattered —
    # 4,476 drawers on 2026-08-06 — so prune before mining, not after.
    prune = (f"find {shlex.quote(dst.mirror_dir)} \\( -name 'subagents' -type d -o -name 'agent-*.jsonl' \\) "
             f"-prune -exec rm -rf {{}} + 2>/dev/null; true"
             if dst.remote else
             f"Get-ChildItem '{dst.mirror_dir}' -Recurse -Directory -Filter subagents -EA SilentlyContinue | "
             f"Remove-Item -Recurse -Force -EA SilentlyContinue; "
             f"Get-ChildItem '{dst.mirror_dir}' -Recurse -File -Filter 'agent-*.jsonl' -EA SilentlyContinue | "
             f"Remove-Item -Force -EA SilentlyContinue")
    dst.sh(prune)

    # Mine, one invocation per project, each under the wing of the machine the
    # conversation actually happened on.
    projects = sorted({f.split("/")[0] for f in todo})
    for proj in projects:
        wing = wing_for(proj)
        path = f"{dst.mirror_dir}/{proj}"
        print(f"    mining {proj} -> wing {wing}")
        env = "PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8"
        pre = f"{dst.pre_mine} && " if getattr(dst, "pre_mine", "") else ""
        if dst.remote:
            dst.sh(f"{pre}{env} {shlex.quote(dst.mempalace)} mine {shlex.quote(path)} "
                   f"--mode convos --cursor --wing {shlex.quote(wing)}")
        else:
            dst.sh(f"$env:PYTHONUNBUFFERED='1'; $env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; "
                   f"& '{dst.mempalace}' mine '{path}' --mode convos --cursor --wing '{wing}'")
    return len(todo)


def main():
    ap = argparse.ArgumentParser(description="Sync two MemPalace machines over SSH.")
    ap.add_argument("--pull-only", action="store_true", help="remote -> here only")
    ap.add_argument("--push-only", action="store_true", help="here -> remote only")
    ap.add_argument("--dry-run", action="store_true", help="report the delta, ship nothing")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--config", default=str(CONFIG))
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        sys.exit(f"no config at {cfg_path}\nCopy palace_sync.config.example.json and fill it in.")
    cfg = json.loads(cfg_path.read_text())
    local, remote = Peer(cfg["local"], False), Peer(cfg["remote"], True)

    # A mirror that does not exist yet is fine; one that is wrong is not, so make
    # the operator say the path rather than letting the script guess.
    for p in (local, remote):
        if not p.mirror_dir or not p.projects_dir:
            sys.exit(f"{p.name}: mirror_dir and projects_dir are both required")

    moved = 0
    if not args.push_only:
        moved += sync(remote, local, "PULL", args)
    if not args.pull_only:
        moved += sync(local, remote, "PUSH", args)
    print(f"\ndone â€” {moved} transcript(s) moved")


if __name__ == "__main__":
    main()

