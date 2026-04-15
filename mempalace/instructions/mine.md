# MemPalace Mine

When the user invokes this skill, follow these steps:

## 1. Ask what to mine

Ask the user what they want to mine and where the source data is located.
Clarify:
- Is it a project directory (code, docs, notes)?
- Is it conversation exports (Claude, ChatGPT, Slack)?
- Do they want auto-classification (decisions, milestones, problems)?

## 2. Choose the mining mode

There are three mining modes:

### Project mining

    mempalace mine <dir>

Mines code files, documentation, and notes from a project directory.

### Conversation mining

    mempalace mine <dir> --mode convos

Mines conversation exports from Claude, ChatGPT, or Slack into the palace.

#### Incremental mining for append-only JSONL (Claude Code, Codex CLI)

For `~/.claude/projects/` or `~/.codex/sessions/`, add `--cursor`:

    mempalace mine <dir> --mode convos --cursor

Claude Code and Codex CLI JSONL transcripts are strictly append-only,
so cursor mode seeks past a stored byte cursor and only processes
newly-appended content. Subsequent mines (on the next hook fire or
manual re-run) skip already-ingested bytes — no re-embedding.

Always safe to include: for any source that isn't recognized as
append-stable (markdown, ChatGPT exports, Slack), cursor mode
silently falls back to the full-mine path. Recommended for any
ongoing `mempalace mine` call against live transcript directories.

### General extraction (auto-classify)

    mempalace mine <dir> --mode convos --extract general

Auto-classifies mined content into decisions, milestones, and problems.
Combine with `--cursor` for incremental general-extraction mining.

## 3. Optionally split mega-files first

If the source directory contains very large files, suggest splitting them
before mining:

    mempalace split <dir> [--dry-run]

Use --dry-run first to preview what will be split without making changes.

## 4. Optionally tag with a wing

If the user wants to organize mined content under a specific wing, add the
--wing flag:

    mempalace mine <dir> --wing <name>

## 5. Show progress and results

Run the selected mining command and display progress as it executes. After
completion, summarize the results including:
- Number of items mined
- Categories or classifications applied
- Any warnings or skipped files

## 6. Suggest next steps

After mining completes, suggest the user try:
- /mempalace:search -- search the newly mined content
- /mempalace:status -- check the current state of their palace
- Mine more data from additional sources
