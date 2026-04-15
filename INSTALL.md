# MemPalace — cursor-patched fork

A local-first memory system for Claude Code. This fork adds **incremental
transcript ingestion** so past conversations get mined once and then only
new content is processed on subsequent runs — instead of re-embedding
every transcript from scratch every time the save hook fires.

If you're here because someone sent you this link, the short version is:
it gives your Claude Code sessions semantic memory across time, runs 100%
locally, and doesn't re-do hours of work whenever the hook fires.

---

## What this fork adds (vs upstream `MemPalace/mempalace`)

See [`CHANGELOG-personal-v2.md`](#changes-from-upstream) below for the
full list. The headline changes:

- **Cursor-based incremental ingest** for Claude Code / Codex CLI JSONL
  transcripts. First mine processes everything; subsequent mines only
  embed newly-appended content.
- **Content-addressed drawer IDs** — re-mining identical content is
  idempotent (same content → same ID, no duplicates, crash-safe).
- **Per-subdirectory wings** — pointing `mempalace mine` at a parent
  directory like `~/.claude/projects/` creates one wing per project slug
  instead of dumping everything into a single "projects" wing.
- **Hook fixes** — Stop hook now counts only real user messages (not
  tool-call echoes) and skips subagent stops (was triggering 3x per save
  cycle).
- **Status command scales** — batched reads so `mempalace status` works
  past 32k drawers (upstream hits SQLite's default variable limit).

All changes are on the `personal-v2` branch. `main` tracks upstream.

---

## Install (Claude Code agent — recommended)

If you're using Claude Code, paste the entire block below into your
session. Your Claude agent will execute it step by step and stop at
anything that needs your attention.

````
Please install this patched MemPalace for me. Follow these steps in order
and stop if any command fails:

1. Verify Python 3.9+ is available:
   python3 --version

2. Clone the fork. Replace REPO_URL with the URL my friend sent:
   git clone REPO_URL ~/code/mempalace-personal
   cd ~/code/mempalace-personal
   git checkout personal-v2

3. Install the package in editable mode (so future git pulls update
   the installed code automatically):
   python3 -m pip install --user --break-system-packages -e ~/code/mempalace-personal

4. Add the user-install bin to PATH (adjust Python version if needed):
   PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
   echo "export PATH=\"\$HOME/Library/Python/$PY_VER/bin:\$PATH\"" >> ~/.zprofile
   source ~/.zprofile

5. Verify the CLI works:
   mempalace --help

6. Register the MCP server with Claude Code at user scope:
   claude mcp add mempalace --scope user -- python3 -m mempalace.mcp_server
   claude mcp list

7. Make the hook scripts executable:
   chmod +x ~/code/mempalace-personal/.claude-plugin/hooks/mempal-stop-hook.sh
   chmod +x ~/code/mempalace-personal/.claude-plugin/hooks/mempal-precompact-hook.sh

8. Configure hooks in ~/.claude/settings.json. Back up first:
   cp ~/.claude/settings.json ~/.claude/settings.json.bak.$(date +%Y%m%d-%H%M%S) 2>/dev/null
   Then merge this into ~/.claude/settings.json (preserve all existing
   keys, add this "hooks" key if absent):
   {
     "hooks": {
       "Stop": [
         {
           "matcher": "*",
           "hooks": [
             {"type": "command", "command": "/Users/USERNAME/code/mempalace-personal/.claude-plugin/hooks/mempal-stop-hook.sh"}
           ]
         }
       ],
       "PreCompact": [
         {
           "matcher": "*",
           "hooks": [
             {"type": "command", "command": "/Users/USERNAME/code/mempalace-personal/.claude-plugin/hooks/mempal-precompact-hook.sh"}
           ]
         }
       ]
     }
   }
   Replace /Users/USERNAME with the actual home directory path.

9. Symlink the slash commands and skill into user scope:
   mkdir -p ~/.claude/skills ~/.claude/commands
   ln -sfn ~/code/mempalace-personal/.claude-plugin/skills/mempalace ~/.claude/skills/mempalace
   for f in ~/code/mempalace-personal/.claude-plugin/commands/*.md; do
     ln -sfn "$f" ~/.claude/commands/mempalace-$(basename "$f")
   done

10. Restart Claude Code. In a new session, type /mcp to confirm
    mempalace is listed with its tools. Type / to confirm the
    /mempalace-* slash commands appear.

If any step fails, stop and tell me the exact error message. Don't
skip or improvise.
````

That's it. Your Claude Code agent will walk through each step and
report back.

---

## Install (manual)

If you'd rather do it yourself or aren't using Claude Code:

```bash
# Prerequisites: Python 3.9+, git, Claude Code CLI (`claude`)

# 1. Clone the fork
git clone REPO_URL ~/code/mempalace-personal
cd ~/code/mempalace-personal
git checkout personal-v2

# 2. Editable install (tracks your local clone — no reinstall on git pull)
python3 -m pip install --user --break-system-packages -e .

# 3. Put the CLI on PATH (macOS, Python 3.13 — adjust version if needed)
echo 'export PATH="$HOME/Library/Python/3.13/bin:$PATH"' >> ~/.zprofile
source ~/.zprofile
mempalace --help   # verify

# 4. Register MCP server (user scope, available in every Claude session)
claude mcp add mempalace --scope user -- python3 -m mempalace.mcp_server

# 5. Make hook scripts executable
chmod +x .claude-plugin/hooks/mempal-stop-hook.sh
chmod +x .claude-plugin/hooks/mempal-precompact-hook.sh

# 6. Edit ~/.claude/settings.json — add a "hooks" key pointing at the
#    two scripts above. See the Claude-Code-agent block above for the
#    exact JSON.

# 7. Symlink the skill and slash commands
mkdir -p ~/.claude/skills ~/.claude/commands
ln -sfn "$(pwd)/.claude-plugin/skills/mempalace" ~/.claude/skills/mempalace
for f in .claude-plugin/commands/*.md; do
  ln -sfn "$(pwd)/$f" ~/.claude/commands/mempalace-$(basename "$f")
done
```

Restart Claude Code after step 7 for the skill/commands to be discovered.

---

## First use

### Mine your past Claude Code conversations

Open a new Claude session and ask:

> "Please mine all my past Claude Code conversations by running:
> `mempalace mine ~/.claude/projects --mode convos --cursor`"

Or run it directly in a terminal:

```bash
PYTHONUNBUFFERED=1 nohup mempalace mine ~/.claude/projects --mode convos --cursor \
  > /tmp/mempalace-mine.log 2>&1 &

# Watch progress in another terminal
tail -f /tmp/mempalace-mine.log
```

Runtime depends on how much history you have. A few megabytes: seconds.
Several gigabytes of transcripts: 30 minutes to an hour for the initial
pass (one-time — subsequent mines are incremental). The first mine also
downloads a ~80 MB embedding model from HuggingFace (one-time).

### Check what was mined

```bash
mempalace status
```

Shows wings (one per project) and drawer counts per room.

### Search

```bash
mempalace search "why did we switch to GraphQL"
```

Or from within Claude:

> "Search mempalace for everything we've discussed about the auth migration."

The AI calls `mempalace_search` via MCP and returns verbatim chunks from
your past conversations.

---

## What runs automatically

Once installed, two hooks fire during every Claude Code session:

**Stop hook** — every 15 user messages, triggers:
1. A background `mempalace mine --cursor` on the active session's
   transcript directory. Captures the raw conversation verbatim.
2. A `decision: block` message telling the AI to save a curated
   summary via `mempalace_diary_write` and key quotes via
   `mempalace_add_drawer`.

**PreCompact hook** — fires synchronously right before Claude Code
compacts your context window. Same save flow, blocking this time so
memories land before details get compressed.

You don't run these manually. They're invisible while working —
only every 15 turns does the AI surface the save prompt.

---

## Troubleshooting

### "command not found: mempalace"

The user-install bin directory isn't on your PATH. Find it:

```bash
python3 -m site --user-base
# e.g. /Users/YOU/Library/Python/3.13
```

Add `<that-path>/bin` to PATH in `~/.zprofile` or `~/.bashrc`, then
`source` the file or open a new terminal.

### MCP server shows "Disconnected" in /mcp

Verify mempalace is importable by the same Python that `claude mcp add`
used:

```bash
which python3
python3 -c "import mempalace; print(mempalace.__file__)"
```

Should print the path to your clone. If not, the editable install
failed. Re-run step 3 of the install.

### "No palace found" when searching

The palace is created automatically on the first mine or MCP write.
If you've never mined anything, there's nothing to search. Run:

```bash
mempalace mine ~/.claude/projects --mode convos --cursor
```

### Hook fires 3x per save

Confirm you're on `personal-v2` branch:

```bash
cd ~/code/mempalace-personal
git branch --show-current   # should print "personal-v2"
```

If it prints `main`, check out the branch:

```bash
git checkout personal-v2
```

### "too many SQL variables" on `mempalace status`

Your palace has crossed 32,766 drawers. This fork fixes that — make
sure you're on `personal-v2`. If you're on this fork and still seeing
it, the fix is at commit `ba8b401`; check your `git log --oneline |
head -15` for that hash.

### The mine takes forever

Large Claude Code archives (1+ GB of JSONL) can take 30-60 minutes for
the initial embed. Watch progress:

```bash
tail -f /tmp/mempalace-mine.log
```

If the log stops growing for 5+ minutes, there may be an issue. Check
the process:

```bash
pgrep -fl mempalace
```

---

## Safety / what this touches

- **Creates** `~/.mempalace/` (your palace data — embeddings + SQLite)
- **Creates** `~/.cache/chroma/` (downloaded embedding model, ~92 MB)
- **Edits** `~/.claude/settings.json` (adds hooks — back up first)
- **Edits** `~/.zprofile` (adds PATH entry)
- **Creates** `~/.claude/skills/mempalace` symlink
- **Creates** `~/.claude/commands/mempalace-*.md` symlinks

No network calls after the initial embedding model download. No API
keys required. No data leaves your machine.

---

## Changes from upstream

Branch `personal-v2` adds 11 commits on top of upstream `main`:

| Commit type | Summary |
|---|---|
| `test` | Failing-test spec for cursor-based incremental ingest |
| `feat(storage)` | Content-addressed drawer IDs + NORMALIZE_VERSION 3 bump |
| `feat(cursor)` | Cursor primitives: `_is_cursor_eligible`, `_find_safe_boundary`, `_read_cursor`, `_write_cursor` |
| `feat(cursor)` | `_mine_with_cursor` + `_normalize_tail` — the cursor-aware mine itself |
| `feat(cursor)` | Wire cursor mode into `mine_convos` + add `--cursor` CLI flag |
| `feat(hooks)` | Wire cursor-aware auto-ingest into Stop and PreCompact |
| `docs(skill)` | Teach mine instruction about `--cursor` flag |
| `fix(hooks)` | Skip Stop hook for subagent completions |
| `fix(hooks)` | Don't count `tool_result` lines as user messages |
| `fix(convos)` | Derive wing per-subdirectory when no `--wing` override |
| `fix(status)` | Batch drawer reads to scale past 32k SQLite var limit |

Run `git log --oneline main..personal-v2` for full commit history.

---

## License

Same as upstream: MIT. See `LICENSE`.

Upstream project: [github.com/MemPalace/mempalace](https://github.com/MemPalace/mempalace)

Issues with the fork's patches: file against the fork repo. Issues
with the core product (architecture, design, base features): file
upstream.
