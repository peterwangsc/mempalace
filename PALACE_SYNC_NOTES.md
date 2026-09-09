# Two-machine palace operations

Peter runs two palaces: the PC (`THUNDERBONE-V3`, `~/.mempalace`) and the Mac
(`~/.mempalace`). The PC's was seeded from the Mac on 2026-07-16 and has forked
since. This note is what the two machines have to agree on. It is deliberately
untracked scratch — see the end for why.

Repo locations differ: `~/code/mempalace` on the PC, `~/code/playground/mempalace`
on the Mac. Both on `personal-v3`, sharing the `peterwangsc/mempalace` remote —
which is how the sync tooling itself travels between them. Keep both checkouts on
the same commit before running a sync; a machine running the older script has a
different idea of what to ship.

## Wings encode origin, not project alone

```
_users_peterwang_code_<project>   Mac-origin transcripts
c__users_pewa_code_<project>      PC-origin transcripts
```

One wing per project *per machine*. This is a deliberate convention, not drift.
Wing is the only filterable axis that records which machine a conversation
happened on, and keeping the split costs nothing at recall time because searches
run unfiltered by default — verified 2026-07-27, unfiltered results interleave
both wings ranked purely by similarity.

Always pass `--wing` explicitly with the **origin** wing of the transcripts being
mined. Never "whichever wing has the most drawers."

Genuinely degenerate wings are a separate matter and are worth deleting: a bare
project name holding one drawer (`golfcore`, `tmp`), or the session-UUID wings.

## `source_file` is a full absolute path

```
drawer_id = drawer_{wing}_{room}_{sha256(source_file + chunk_content)[:24]}
cursor_id = _reg_{sha256(source_file)[:24]}
```

`source_file` stores the **full absolute path**, not the basename. `mempalace_search`
*displays* a bare filename, which makes it easy to conclude otherwise — read raw
Chroma metadata if you need to be sure.

Three consequences, in descending order of how much they will hurt you:

1. **The mirrors must never move.** `~/mac-transcripts/` on the PC and
   `~/pc-transcripts/` on the Mac are permanent addresses. Relocating or
   re-rooting either one changes every `source_file`, which resets every cursor
   and refiles the entire corpus as duplicate drawers under fresh IDs.
2. **The two palaces never hold identical drawer IDs** for the same conversation.
   They are not convergent replicas; they are two indexes over overlapping
   corpora. Do not write tooling that assumes ID equality across machines.
3. **Re-mining is idempotent only per machine per path.** Same file from the same
   location upserts harmlessly. Same file from a second location duplicates every
   drawer in it.

Renaming a wing does the same damage as moving a mirror, and for the same reason:
the wing is baked into the drawer ID.

## Sync procedure — one command, from either side

### Claude and Codex sources (2026-09-09)

Both `sync.sh` and `sync.ps1` call `scripts/palace_sync.py`. Existing
`projects_dir` and `mirror_dir` remain the Claude source and permanent mirror.
To include Codex, configure **both peers** with `codex_projects_dir` pointing at
their `~/.codex/mempalace-transcripts` hook exports and `codex_mirror_dir` pointing
at a separate permanent mirror: `C:/Users/pewa/mac-codex-transcripts` on PC and
`/Users/peterwang/pc-codex-transcripts` on Mac. Never nest it in the Claude mirror.
The default sync includes all configured sources; `--source codex` or
`--source claude` limits a run. Old configurations still sync Claude with an
explicit notice that Codex is not configured. Partial Codex configurations fail.

Codex exports already group by canonical machine-origin project wing. The same
wing rule applies to both formats. Raw `sessions/YYYY/MM/DD` trees and SQLite
databases are not sync sources. Hooks create exports as sessions progress;
older sessions need a one-time adapter backfill before they can sync. This sync
does not backfill unexported history or copy the vector database.

Sync refuses to overwrite a longer mirror transcript with a shorter source.
Equal sizes still mean unchanged under the existing append-only assumption.
Coordinate one sync invocation for the pair; the script is not a concurrent
bidirectional scheduler. Verify both checkouts/configurations before running it.

Transcripts are the source of truth; the vector DB is derived. Never attempt to
merge `chroma.sqlite3` files — there is no import path (`exporter.py` emits lossy
markdown only), and hand-editing Chroma internals has corrupted this palace before.

`scripts/palace_sync.py` does the whole thing, both directions, and the bridge is
symmetric — run it from whichever machine you are sitting at. The shell you are in
does not have to be the one holding the files.

```shell
./sync.sh                # Mac: pull PC → here, then push here → PC
./sync.ps1               # PC: same, both directions
./sync.sh --dry-run      # report the delta, ship nothing
./sync.sh --pull-only    # or --push-only
./sync.sh -v             # name the first 20 files in each delta
```

Both wrappers set `ulimit -n 10240` where it applies and call
`.venv/bin/python scripts/palace_sync.py`. Never invoke a bare `python`: over SSH
the Mac resolves 3.9.6 and in a login shell 3.14.6, and `npm run sync:*` is
unusable on the Mac because node is not installed there.

### It needs a config you must write once

`scripts/palace_sync.config.json`, gitignored because it describes this machine
pair rather than the software. Copy `palace_sync.config.example.json`. `local` is
always the machine you are running on, `remote` is the peer, so **the two machines
hold mirror images of each other's file** — the Mac's `local` is the PC's `remote`.
Every path is absolute and permanent; see the `source_file` section above for why
`mirror_dir` in particular can never move.

### What it does, in order

1. Inventories both sides — `{relative path: size}` for everything under
   `projects_dir`, and the same for the peer's `mirror_dir`.
2. Takes the delta by size. Transcripts are append-only, so a size match means
   nothing new; the mirror *is* the materialised cursor. Ships only what is
   missing or has grown.
3. Tars the delta from an explicit file list, so nothing outside it can ride along.
4. `scp`s it, then compares SHA-256 on both ends and aborts before extracting on a
   mismatch — a truncated archive extracts to a plausible-looking subset.
5. Extracts into the permanent mirror, deleting `._*` AppleDouble stubs.
6. Mines one invocation per project directory, each `--mode convos --cursor` under
   the origin wing derived from the project directory name.

### What it ships

Exactly what `mempalace mine --mode convos` ingests: `.jsonl`, `.txt`, `.md` and
`.json` under `os.walk`, minus `convo_miner.CONVO_SKIP_DIRS`. Session transcripts,
`subagents/*.jsonl` and `memory/*.md` travel; `tool-results/` does not;
`*.meta.json` and `._*` are dropped.

**Keep the sync's walk and `CONVO_SKIP_DIRS` in step.** The hooks are the real
source: Stop and SessionEnd mine `Path(transcript_path).parent` — the whole
project directory — recursively. So whatever the miner takes, the machine of
origin has already filed on every stop. A sync that filters *more* leaves the
mirror holding less than the palace it mirrors; one that filters *less* files
content the origin never did. Neither drifts silently, and both are wrong.

Why `subagents/` stays and `tool-results/` goes, decided 2026-08-06 after
sampling both. Subagent transcripts carry findings — a file path, a refactor
rationale, a security verdict — that the parent transcript only ever recorded as
a summary; they are the leads a summary discards. `tool-results/` holds the raw
bytes a tool returned before anyone reasoned about them: linter output with
timestamps, grep dumps, stray HTML, and — worst — serialised `mempalace_search`
responses, which file the palace's own output back as palace content so later
searches can match a copy of an earlier search.

Measured on the Mac that day, all of it steady-state hook behaviour rather than
anything a tarball did: 81,384 subagent drawers, and 19,932 tool-result drawers
across 17,886 unnamed dumps, 2,084 `toolu_*`, 188 `mcp-mempalace-*` and one
321,773-character image-generation payload. The 19,932 already filed were left in
place; only the inflow was stopped.

### Addressing

**Only ever by ssh alias, never a raw IP.** `thunderbone` is `~/.ssh/config` →
`thunderbone-v3.lan`, user `pewa`; `mac` is the alias in `C:/Users/pewa/.ssh/config`.
Both DHCP leases have moved and silently killed the bridge — the PC's .232 → .236,
the Mac's .21 → .241 on 2026-08-02. Only the alias you dial has to be current, so
driving from the Mac is the safer default: `thunderbone` is the always-working
direction, while `mac` is the one that has gone stale and eaten a night of note
pushes. The Mac owns keeping it current — check `ipconfig getifaddr en0` and
rewrite the PC's config over `thunderbone`. The bridge itself (two-way,
key-authenticated, set up 2026-07-16) is documented in golfcore's
`script/bake/PROTOCOL.md`.

### Doing it by hand

Only if the script is unavailable. Tar the delta from `projects_dir` with relative
paths, `scp` it, extract with `-C <mirror_dir>`, `find <mirror> -name '._*' -delete`,
then one `mempalace mine <mirror>/<project> --mode convos --cursor --wing <origin>`
per project. Getting the mirror path or the wing wrong duplicates every drawer in
every file, which is the whole reason the script refuses to guess either.

## Gotchas

- **macOS `ulimit -n` is 256.** Chroma's `_ensure_fresh()` reopens a
  `PersistentClient` repeatedly and blows past it mid-mine, dying with
  `pyo3_runtime.PanicException: ... Os { code: 24, ... "Too many open files" }`.
  Run `ulimit -n 10240` first. The hard limit is unlimited.
- **macOS `tar` emits `._` AppleDouble stubs** that match `*.jsonl` globs and get
  fed to the miner as garbage. `find <dir> -name '._*' -delete` after extracting.
- **Mac project directories start with `-`** (`-Users-peterwang-code-golfcore`), so
  a bare `find <dir> ...` parses the name as a flag and silently matches nothing —
  no error, just zero results. Always `find ./<dir>`.
- **Redirected mine output is block-buffered.** A crash can leave a 0-byte log and
  look like a silent death. Set `PYTHONUNBUFFERED=1` when logging to a file.
- Cursors live *inside* the palace as `_reg_*` sentinel drawers, not on the
  filesystem, so each palace tracks its own progress independently.

## This file is tracked on `personal-v3`

It was untracked until 2026-08-06, on the reasoning that it describes this pair of
machines rather than the software. Hand-mirroring turned out to be the weaker
half of that trade: the PC ran a version of `palace_sync.py` for an hour while
holding the pre-sync copy of these notes, and nothing warns you when the two
drift. On a personal branch it travels by git instead.

What still must never travel: `scripts/palace_sync.config.json`, which stays
gitignored. Structure lives in `palace_sync.config.example.json`; the aliases,
absolute paths and machine names live only in the real file.

Do not carry this file into a PR to `MemPalace/mempalace` upstream — it names
hosts, home directories and DHCP history that belong to these two machines.

_Last synced 2026-08-06. PC→Mac: 9 new transcripts, `c__users_pewa_code_golfcore`
5,580 → 8,750, `c__users_pewa_code_golfcore_site` 0 → 112. Mac palace at 252,036
drawers. Mac→PC shipped by hand that day and had to be repaired on the PC: the
archive was filtered by extension only, so 124 nested `subagents/agent-*.jsonl`
rode along and cost 4,476 drawers there. That is the incident `palace_sync.py`
exists to make impossible._

_Prior: 2026-07-27. Mac→PC 93 transcripts, golfcore wing 36,562 → 59,186.
PC→Mac 29 transcripts, `c__users_pewa_code_golfcore` 0 → 5,580._
