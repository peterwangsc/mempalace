---
name: mempalace
description: MemPalace — the memory palace of prior decisions, and the chained-search method for investigating with it. Use when asked about mempalace, memory palace, mining or searching memories, palace setup, and whenever an investigation spans more than one file, symbol, or decision.
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# MemPalace

A searchable record of what was decided and why. The code says what is true
now; the palace says what was true before, what was tried, what it cost, and
which constraint is load-bearing. Most investigations need both.

## The method: chained search

This is the part that matters. A single search answers a lookup. An
investigation needs a chain — and the chain is the skill.

**1. Decompose the ask into concepts, not questions.**
A question is one string. An investigation is several nouns. "Why is the page
slow and why does the socket reconnect" is not one search; it is `payload size
asset fetch`, `socket subscription lifecycle`, `render gate mount order`. Name
each concept before searching for any of them.

**2. Fire a round of 3-5 distinct queries in one parallel block.**
Keep each `query` short and keyword-dense — only that field is embedded.
Different phrasings of the same concept are wasted slots; different concepts
are not.

**3. Read results for leads, not for answers.**
This is the step that is usually skipped. The value of a result is rarely the
prose. It is a filename you did not know existed, an exact symbol, a measured
number, a phrase in the project's own vocabulary, a decision with a date. Note
every one. A result that does not answer your question but names a file you
have not read is a *good* result.

**4. Turn each lead into the next round's query.**
This is the chain. Round two searches the vocabulary round one taught you. Round
three searches what round two revealed. Expect the useful query to be one you
could not have written at the start, because it uses a term you had not yet
learned.

**5. Alternate search and reading. Do not batch them.**
query → read → query → read. Reading a file mid-chain is not a detour; it is
what makes the next query specific. The palace points you at the file; the file
tells you what to search next. Either one alone stalls.

**6. Verify palace claims against the running system.**
Drawers are historical. A recorded rationale can be obsolete — the dependency
was upgraded, the code moved, the guard now exists upstream. When a drawer
supplies the *reason* for something load-bearing, open the source and confirm
the reason still holds. Report it when it does not; a stale rationale is often
the most valuable thing in the investigation.

**7. Measure what the palace merely states.**
Recorded numbers are a starting point and a cross-check, never the answer.
Re-measure live where the number drives a decision.

**8. Stop when the question yields, not when you have enough to write.**
An empty result indicts the query, not the palace. Rephrase — keyword form,
question form, the error string verbatim, the domain noun — and conclude
absence only after several distinct phrasings return nothing.

**Expect eight to ten rounds on a real investigation.** Fewer usually means you
stopped at the first plausible answer. There is no search budget; searches are
cheap and being wrong in front of the user is not.

## Do not delegate the chain

Never hand a chained investigation to a subagent. A subagent returns a summary,
and a summary is exactly the thing that destroys a chain: the leads — the stray
filename, the odd number, the unfamiliar term — are discarded as noise on the
way to a tidy conclusion, and those leads were the whole mechanism. The chain
has to run in one context because every result reshapes the next query.

Subagents are for breadth with no follow-up: sweeping many files for one fixed
pattern, or independent work with a known shape. The moment the next question
depends on this answer, keep it.

## Reporting

State the measured value, the exact identifier, and what you could not verify.
Separate what you confirmed against the running system from what you inferred.
When you correct an earlier claim of your own, say so in one sentence and move
on.

## Writing back

Write reusable technical facts back so the next chain starts further along:
`mempalace_diary_write` for decisions and reasoning,
`mempalace_kg_add` for atomic facts. Supersede rather than duplicate; delete
what proves wrong. Record the fact that prevents the error, not the error.

## Mechanics

Prefer the MCP tools: `mempalace_search` first, then `mempalace_kg_query` for
atomic facts, `mempalace_diary_read` for reflections, `mempalace_traverse` and
`mempalace_find_tunnels` for relations. Their live descriptions are the
authority on arguments; do not restate them here.

Search hygiene: only `query` is embedded, so keep it short; `context` is not
embedded and text there is wasted; `max_distance` defaults to 1.5, and 0.6-0.8
narrows to a near-exact memory. Wing filtering is unreliable — search globally
and narrow only if the results demand it, and on an empty wing-filtered result
retry with no filter.

**Do not run `mempalace instructions <command>`.** It emits a generic setup
wizard written for a single machine — ask-the-user prompts, next-step menus, a
stale tool count, and a search procedure that tells you to filter by wing,
which contradicts the line above. Its five topics cover four of the CLI's
fifteen subcommands, omitting `sweep`, `wake-up`, `compress`, `repair` and
`repair-status`. Read `mempalace <command> --help` for real flags.

How this palace is mined, synced and repaired — which wing a transcript belongs
to, what must never be re-rooted, what is excluded from mining, and whether any
of it is automatic — is machine-specific and lives in the operating manual.
That file governs. Nothing here restates it, so the two cannot drift apart.

Installation, if missing: `pip install mempalace`.
