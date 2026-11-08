---
name: mempalace
description: MemPalace — the memory palace of prior decisions, and the chained-search method for investigating with it. Use when asked about mempalace, memory palace, mining or searching memories, palace setup, and whenever an investigation spans more than one file, symbol, or decision.
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# MemPalace

The code says what is true now. The palace says what was decided before, what
was tried, what it cost, and why. Search it whenever the work turns on a past
decision. A question about why data or code looks the way it does is usually
answered there.

Search before you build anything to find something out: a script, a query, a
probe, an API call. On 2026-08-17 a deliberate placeholder was hand-patched on
production as a "bug", and forty minutes went into rebuilding an approach the
palace recorded as already tried and dropped. Two searches would have found it.

Every new concept that comes up mid-task is its own search. One round at the
start of a session does not cover the rest of it.

## The method: chained search

1. **Split the ask into concepts.** Name each noun before searching any of
   them.
2. **Fire 3-5 distinct queries in one parallel block.** Short and
   keyword-dense. Different concepts, not rephrasings of one.
3. **Read results for leads, not answers.** A filename, a symbol, a number, a
   dated decision, a term in the project's own words. A result that names a
   file you have not read is a good result.
4. **Each lead becomes the next round's query.** The useful query uses a term
   you did not know at the start.
5. **Alternate searching and reading.** The palace points to the file. The file
   tells you what to search next.
6. **Check palace claims against the running system.** A recorded reason can
   be stale. Report it when it is.
7. **Re-measure numbers the palace states** when they drive a decision.
8. **Stop when the question yields.** An empty result indicts the query.
   Conclude absence only after several distinct phrasings return nothing.

Expect eight to ten rounds on a real investigation. Searches are cheap.

Never hand a chain to a subagent. Its summary throws away the leads, and the
leads are the mechanism. Subagents suit fixed-pattern sweeps with no follow-up.

## Reporting and writing back

Report the measured value, the exact identifier, and what you could not
verify. Separate what you confirmed from what you inferred.

Do not write to the palace unless Peter asks in that turn. The session hooks
are the only ingest path. Anything that must be seen every session goes in a
file that loads every session.

## Tools

Use `mempalace_search` first, then `mempalace_kg_query` for atomic facts,
`mempalace_diary_read` for reflections, and `mempalace_traverse` /
`mempalace_find_tunnels` for relations. Their live descriptions are the
authority on arguments.

- Only `query` is embedded. `context` is not, so text there is wasted.
- `max_distance` defaults to 1.5. Use 0.6-0.8 for a near-exact memory.
- Wing filtering is unreliable. Search unfiltered. On an empty filtered
  result, retry with no filter.
- Never run `mempalace instructions <command>`. It emits a stale single-machine
  wizard that contradicts this file. Use `mempalace <command> --help`.
- If missing: `pip install mempalace`.

## The PC (THUNDERBONE-V3)

Both machines load this file. These lines describe the PC. On the Mac, its
operating manual governs mining and sync.

- **Mining is automatic.** `~/.claude/settings.json` registers `Stop` and
  `SessionEnd` hooks running `mempalace hook run`. A manual `mine` is only for a
  corpus the hooks never saw, such as a mirror synced from the Mac. Never run
  one to "save" the current session.
- **The palace forked from the Mac's on 2026-07-16.** Older drawers carry
  `/Users/peterwang/...` paths. The Mac's later sessions never appear here, so
  never conclude "we never did X" from an absence. The palaces converge by
  syncing raw transcripts and re-mining, never by merging ChromaDB files.
- **Wings encode origin.** `_users_peterwang_code_<project>` is Mac-origin,
  `c__users_pewa_code_<project>` is PC-origin. Mine with `--wing` set to the
  origin wing, never the largest one.
- **Manual mine:**
  `mempalace mine "$HOME\.claude\projects\<slug>" --mode convos --cursor --wing <wing>`.
  The slug is the working directory with `\` and `:` replaced by `-`.
- **Call it as `mempalace`.** The shim `~/.local/bin/mempalace.cmd` sets the
  UTF-8 variables the executable needs. The venv `Scripts` directory stays off
  `PATH`; it would shadow anaconda's `python.exe` and `pip.exe`.
- **Never re-root a mirror or rename a wing.** Drawer ids hash the full
  absolute source path, so mining one transcript from two paths duplicates
  every drawer. The mirrors are fixed: `~/mac-transcripts/` here,
  `~/pc-transcripts/` on the Mac.
- **`--cursor` is always safe.** There is one cursor per transcript file, not
  per wing, so a re-mine into another wing skips. Reset the cursor to force a
  full pass.
- **`mempalace sweep <file-or-dir>`** files one verbatim drawer per message and
  is what recall is built on. It is idempotent and takes no `--wing` or
  `--cursor`.
- **Subagent transcripts are mined on purpose** into UUID-named wings. Never
  purge a wing for being UUID-named.
- **A drawer total above expectation indicts duplication first.** Measure the
  named and UUID wings, and delete only after finding the same content under
  two paths.
