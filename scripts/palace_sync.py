#!/usr/bin/env python3
"""Two-machine palace sync over SSH. Run from either side; it does both directions.

    python scripts/palace_sync.py                 # pull then push
    python scripts/palace_sync.py --pull-only
    python scripts/palace_sync.py --dry-run

Transcripts are the source of truth and the vector DB is derived, so this ships
transcript files and re-mines. It never touches chroma.sqlite3 - there is no
import path and hand-editing has corrupted a palace before.

Both peers are driven through a POSIX shell: the PC reaches Git Bash via the
`shell` key, since its OpenSSH hands commands to cmd.exe and its `bash.exe` on
PATH is a dead WSL stub. There is no PowerShell or cmd syntax anywhere here.

Every rule below is here because breaking it cost real damage once:

  mine parity       Ship exactly what `mempalace mine --mode convos` ingests:
                    an os.walk taking .jsonl/.txt/.md/.json, minus CONVO_SKIP_DIRS.
                    The Stop and SessionEnd hooks already file that same set on
                    the machine of origin, so a sync that filtered more would
                    leave the mirror holding less than the palace it mirrors, and
                    filtering less would file what the origin never did. Subagent
                    transcripts travel; tool-results do not; *.meta.json and ._*
                    are dropped. Keep this walk and CONVO_SKIP_DIRS in step.
  delta only        A full ship is 646 MB and minutes of CPU the cursor then
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
# The set the convo miner itself ingests (convo_miner.CONVO_EXTENSIONS). The
# sync ships exactly what a mine would pick up, so a synced machine and the
# machine of origin file the same drawers: session transcripts, subagent
# transcripts, tool-results/*.txt and memory/*.md alike.
EXTENSIONS = (".jsonl", ".txt", ".md", ".json")

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

DEFAULT_EXCLUDE = []


def scan(root: Path, skip=()):
    """{'<project>/<relative path>': size} — must mirror MANIFEST_PROG exactly,
    or the delta re-ships every run against a mirror that under-reports."""
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
    """`-Users-peterwang-code-golfcore` -> `_users_peterwang_code_golfcore`,
    `C--Users-pewa-code-golfcore` -> `c__users_pewa_code_golfcore`."""
    return project_dir.lower().replace("-", "_")


def run(cmd, **kw):
    """check=True with the far side's stderr attached to the failure.

    Scripts cross as bytes: text mode on Windows rewrites every \\n in `input` to
    \\r\\n, and a POSIX shell then reads a tar target named `x.list\\r`.

    CalledProcessError prints only the argv, and for an ssh call that argv is
    `bash -s` — the actual command and its error live in the payload and the
    stream, so a bare traceback says nothing about what went wrong.
    """
    kw["input"] = kw["input"].encode() if "input" in kw else None
    p = subprocess.run(cmd, capture_output=True, **kw)
    p.stdout, p.stderr = (s.decode("utf-8", "replace") for s in (p.stdout, p.stderr))
    if p.returncode:
        raise SystemExit(f"exit {p.returncode} from {cmd[0]}\n"
                         f"--- stdout ---\n{p.stdout[-4000:]}\n"
                         f"--- stderr ---\n{p.stderr[-4000:]}")
    return p


class Peer:
    """One machine. Every snippet below is POSIX sh, on both sides.

    The PC has no usable WSL and its OpenSSH hands commands to cmd.exe, but Git
    Bash is a full MSYS shell with cat/tar/find/rm and it accepts the config's
    `C:/...` paths. So a peer declares how to reach *a POSIX shell* — `shell` in
    the config — and the script never emits PowerShell or cmd syntax at all.

    Scripts travel on stdin, which is why every payload (the manifest program,
    the tar file list) rides in a heredoc rather than being piped separately.
    """

    def __init__(self, cfg, remote):
        self.__dict__.update(cfg)
        self.remote = remote
        self.shell = cfg.get("shell", "bash -s")
        # GNU tar reads `C:/x` as host:path and tries to rsh to "C". MSYS tar
        # takes --force-local; bsdtar on macOS rejects the flag outright, so it
        # is conditional on the paths actually being Windows ones. Switching this
        # peer to MSYS `/c/...` paths instead is not an option: they land in
        # `source_file` at mine time and would refile the whole corpus.
        self.tar_opt = "--force-local " if ":" in f"{self.tmp_dir}{self.mirror_dir}" else ""

    def sh(self, script):
        if self.remote:
            return run(["ssh", "-o", "BatchMode=yes", self.ssh, self.shell], input=script)
        return run(shlex.split(self.shell), input=script)

    def tar_delta(self, tar: str, files: list):
        """Archive exactly `files`, relative to this peer's projects_dir."""
        body = "\n".join("./" + f for f in files)
        self.sh(f"set -e\ncd {shlex.quote(self.projects_dir)}\n"
                f"cat > {shlex.quote(tar)}.list <<'LIST'\n{body}\nLIST\n"
                f"tar {self.tar_opt}-czf {shlex.quote(tar)} -T {shlex.quote(tar)}.list\n"
                f"rm -f {shlex.quote(tar)}.list\n")

    def tar_count(self, tar: str) -> int:
        out = self.sh(f"tar {self.tar_opt}-tzf {shlex.quote(tar)} 2>/dev/null | wc -l")
        return int(out.stdout.strip() or 0)

    def sha256(self, path: str) -> str:
        out = self.sh(f"{shlex.quote(self.python)} - {shlex.quote(path)} <<'PYEOF'\n"
                      "import hashlib, sys\n"
                      "print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())\n"
                      "PYEOF\n")
        return out.stdout.strip().lower()

    def _remote_scan(self, root, skip=()):
        r = self.sh(f"{shlex.quote(self.python)} - {shlex.quote(root)} "
                    f"{shlex.quote('|'.join(skip))} <<'PYEOF'\n{MANIFEST_PROG}\nPYEOF\n")
        return json.loads(r.stdout)

    def manifest(self):
        """{'<project>/<relative path>': size} for everything a mine would ingest."""
        skip = getattr(self, "exclude", None) or DEFAULT_EXCLUDE
        if self.remote:
            return self._remote_scan(self.projects_dir, skip)
        return scan(Path(self.projects_dir), skip)

    def mirror_manifest(self):
        """Same shape, for the copy of the *other* machine's transcripts held here."""
        if self.remote:
            return self._remote_scan(self.mirror_dir)
        return scan(Path(self.mirror_dir))


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
    if args.verbose:
        for f in todo[:20]:
            print(f"      {'NEW ' if f in new else 'GROW'} {f}")
        if len(todo) > 20:
            print(f"      ... {len(todo) - 20} more")
    if args.dry_run:
        print("    (dry run, stopping here)")
        return 0
    if todo:
        ship(src, dst, todo)
    else:
        print("    nothing new to ship")

    # Mine every project in the mirror, not just what moved on this run. Shipping
    # and mining are separate failures: an interrupted mine leaves the mirror
    # complete and the palace short, and a delta computed from the mirror is then
    # empty forever after — the run exits 0 with eight projects never filed. That
    # happened on 2026-08-06. The cursor and the already-filed check make the
    # re-scan cheap; correctness here is worth more than the seconds it costs.
    projects = sorted({k.split("/")[0] for k in dst.mirror_manifest()})
    for proj in projects:
        wing = wing_for(proj)
        path = f"{dst.mirror_dir}/{proj}"
        print(f"    mining {proj} -> wing {wing}")
        env = "PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8"
        pre = f"{dst.pre_mine} && " if getattr(dst, "pre_mine", "") else ""
        dst.sh(f"{pre}{env} {shlex.quote(dst.mempalace)} mine {shlex.quote(path)} "
               f"--mode convos --cursor --wing {shlex.quote(wing)}")
    return len(todo)


def ship(src: Peer, dst: Peer, todo: list) -> None:
    """Archive the delta, move it, verify it, and extract into the mirror."""
    stamp = os.environ.get("PALACE_SYNC_STAMP", "sync")
    tar_name = f"palace-sync-{src.name}-to-{dst.name}-{stamp}.tar.gz"
    src_tar = f"{src.tmp_dir}/{tar_name}"
    dst_tar = f"{dst.tmp_dir}/{tar_name}"

    # Build the archive from an explicit file list: nothing but the delta can end
    # up inside it.
    src.tar_delta(src_tar, todo)

    held = src.tar_count(src_tar)
    if held < len(todo):
        raise SystemExit(
            f"{tar_name} holds {held} entries for a {len(todo)}-file delta - "
            f"refusing to ship. sha256 cannot catch this: an empty archive "
            f"transfers faithfully and matches on both sides.")

    # Move it. scp needs one local and one remote endpoint.
    if src.remote and not dst.remote:
        run(["scp", "-q", f"{src.ssh}:{src_tar}", dst_tar])
    elif dst.remote and not src.remote:
        run(["scp", "-q", src_tar, f"{dst.ssh}:{dst_tar}"])
    else:
        raise SystemExit("one side must be local")

    # Integrity, because a truncated archive extracts to a plausible-looking subset.
    if src.sha256(src_tar) != dst.sha256(dst_tar):
        raise SystemExit(f"checksum mismatch shipping {tar_name} - aborting before extract")
    print(f"    shipped {tar_name}, sha256 verified")

    # Extract into the permanent mirror. Never anywhere else.
    dst.sh(f"set -e\nmkdir -p {shlex.quote(dst.mirror_dir)}\n"
           f"tar {dst.tar_opt}-xzf {shlex.quote(dst_tar)} -C {shlex.quote(dst.mirror_dir)}\n"
           f"find {shlex.quote(dst.mirror_dir)} -name '._*' -delete\n"
           f"rm -f {shlex.quote(dst_tar)}\n")

    # Nothing is pruned from the mirror. The hooks on the machine of origin mine
    # its whole project tree — subagents, tool-results and memory included — so a
    # mirror that dropped them would hold strictly less than the palace it mirrors.
    # `._*` goes because macOS tar emits AppleDouble stubs that match *.jsonl.


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

