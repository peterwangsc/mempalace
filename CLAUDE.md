# CLAUDE.md

## The Mission

Memory is identity. When an AI forgets everything between conversations, it cannot build real understanding — of you, your work, your people, your life.

MemPalace exists to solve this. It is a memory system — not a search engine, not a RAG pipeline, not a vector database wrapper. It treats every word you have shared as sacred, stores it verbatim, and makes it instantly available. Your data never leaves your machine. We never summarize. We never paraphrase. We return your exact words.

100% recall is the design requirement — the target every search path is measured against. Anything less means forgetting, and forgetting means starting over.

The name comes from the ancient "method of loci" — the memory palace technique used for thousands of years to organize and recall vast amounts of information by placing it in imagined rooms of an imagined building. We were also inspired by the Zettelkasten method (created by German sociologist Niklas Luhmann) — small cross-referenced index cards that point to each other. We apply both ideas to AI memory:

- **Wings** for broad categories (people, projects, topics)
- **Rooms** for time-based groupings (days, sessions)
- **Drawers** for full verbatim content (your exact words)
- **AAAK compression** for the index layer — a compact symbolic format (via `dialect.py`) that lets an LLM scan thousands of entries instantly and know exactly which drawer to open

## Design Principles

These are non-negotiable. Every PR, every feature, every refactor must honor them.

- **Verbatim always** — Never summarize, paraphrase, or lossy-compress user data. The system searches the index and returns the original words. If a user said it, we store exactly what they said. This is the foundational promise.
- **Incremental only** — Append-only ingest after initial build. Never destroy existing data to rebuild. A crash mid-operation must leave the existing palace untouched.
- **Entity-first** — Everything is keyed by real names with disambiguation by DOB, ID, or context. People matter more than topics.
- **Local-first, zero external API by default** — All extraction, chunking, embedding, and LLM-assisted refinement happens on the user's machine by default, using locally-hosted runtimes (Ollama, LM Studio, llama.cpp, vLLM, unsloth studio, etc.). External providers (Anthropic, OpenAI, Google) are supported via BYOK but are never required and never enabled silently. The system never sends user content to a service the user has not explicitly configured. "Local LLM" is not an external API — Ollama and equivalents running on localhost are part of the user's machine. External BYOK is always a deliberate user choice, never a default and never a silent fallback.
- **Performance budgets** — Hooks under 500ms. Startup injection under 100ms. Memory should feel instant.
- **Privacy by architecture** — The system physically cannot send your data because it never leaves your machine. No telemetry, no phone-home, no external service dependencies for core operations.
- **Background everything** — Filing, indexing, timestamps, and pipeline work happen via hooks in the background. Nothing interrupts the user's conversation. Zero tokens spent on bookkeeping in the chat window.

## Contributing

We welcome bug fixes, performance improvements, new language support, better entity disambiguation, documentation, and test coverage.

We do not accept summarization of user content, cloud storage/sync features, telemetry or analytics, features requiring API keys for core memory, or shortcuts that bypass verbatim storage.

## Setup

```bash
pip install -e ".[dev]"
```

## Commands

```bash
# Run tests
python -m pytest tests/ -v --ignore=tests/benchmarks

# Run tests with coverage
python -m pytest tests/ -v --ignore=tests/benchmarks --cov=mempalace --cov-report=term-missing

# Lint
ruff check .

# Format
ruff format .

# Format check (CI mode)
ruff format --check .
```

## Project Structure

```
mempalace/
├── mcp_server.py        # MCP server — all read/write tools
├── cli.py               # CLI dispatcher
├── config.py            # Configuration + input validation
├── miner.py             # Project file miner
├── convo_miner.py       # Conversation transcript miner
├── searcher.py          # Semantic search (hybrid BM25 + vector)
├── knowledge_graph.py   # Temporal entity-relationship graph (SQLite)
├── palace.py            # Shared palace operations
├── palace_graph.py      # Room traversal + cross-wing tunnels
├── backends/            # Pluggable storage backends (ChromaDB default)
│   ├── base.py          # Abstract interface — implement this for new backends
│   └── chroma.py        # ChromaDB implementation
├── dialect.py           # AAAK compression dialect
├── normalize.py         # Transcript format detection + normalization
├── entity_detector.py   # Auto-detect people/projects from content
├── entity_registry.py   # Entity storage and disambiguation
├── layers.py            # L0-L3 memory wake-up stack
├── onboarding.py        # Interactive first-run setup
├── repair.py            # Palace repair and consistency checks
├── dedup.py             # Deduplication
├── migrate.py           # ChromaDB version migration
├── spellcheck.py        # Auto-correct user messages
├── exporter.py          # Palace data export
├── hooks_cli.py         # Hook management CLI
├── query_sanitizer.py   # Prompt contamination prevention
├── split_mega_files.py  # Split concatenated transcript files
└── version.py           # Single source of truth for version

hooks/                   # Claude Code hook scripts
├── mempal_save_hook.sh        # Stop: triggers diary save
└── mempal_precompact_hook.sh  # PreCompact: saves state before compression
```

## Conventions

- **Python style**: snake_case for functions/variables, PascalCase for classes
- **Linter**: ruff with E/F/W rules
- **Formatter**: ruff format, double quotes
- **Commits**: conventional commits (`fix:`, `feat:`, `test:`, `docs:`, `ci:`)
- **Tests**: `tests/test_*.py`, fixtures in `tests/conftest.py`
- **Coverage**: 85% threshold (80% on Windows due to ChromaDB file lock cleanup)

### Wing naming — single canonical rule

**Every wing name in the palace must derive from `normalize_wing_name()` applied to a real path component.** No invented `wing_*` namespaces, no hardcoded constants like `wing_session_stub`, no per-tool variants like `wing_claude` vs `wing_claude_code`. The reason is operational: split conventions silently fragment a project's content across parallel wings, so a search for the chandler project's history misses anything that landed in `wing_chandler` instead of `_users_peterwang_code_chandler`.

The rule has one form, applied everywhere:

```python
from mempalace.config import normalize_wing_name
wing = normalize_wing_name(<path-component>)   # lowercase + ' '/'-' → '_'
```

What `<path-component>` is depends on the call site:

- **Convo mining (`convo_miner.mine_convos`)**: `Path(convo_dir).name` for top-level `--wing` derivation, or per-subdir name when scanning a parent of project dirs.
- **Hook flow (`hooks_cli`)**: `Path(transcript_path).parent.name` — the encoded project folder Claude Code creates under `~/.claude/projects/-Users-…-<project>/`. This produces `_users_<user>_..._<project>`, which is identical to what a CLI mine of that same directory would produce.
- **Project mining (`miner.mine`)**: the project root's directory name.
- **Synthetic / stub writes (SessionEnd stubs, diary checkpoints)**: derived from the same source they describe — usually the transcript path. **Never** invent a new wing for a synthetic entry; route it to the wing the actual content lives in.

When adding a code path that creates a wing:

1. Identify the path component the wing represents.
2. Run it through `normalize_wing_name`.
3. If you need a fallback for "no path available," prefer `_sessions` (canonical-prefixed) over `wing_sessions` (legacy/divergent). Never invent a new namespace.

Reviewing PRs that introduce new wing-creating code: reject any new string-template wing name (`f"wing_{x}"`, `"sessions"`, etc.) that isn't ultimately the output of `normalize_wing_name`.

## Architecture

```
User → CLI / MCP Server → Storage Backend (ChromaDB default, pluggable)
                        → SQLite (knowledge graph)

Palace structure:
  WING (person/project)
    └── ROOM (day/topic)
          └── DRAWER (verbatim text chunk)

Index layer (AAAK):
  Compressed pointers → DRAWER locations
  Scanned by LLM to find relevant drawers without reading all content

Knowledge Graph:
  ENTITY → PREDICATE → ENTITY (with valid_from / valid_to dates)
```

## Key Files for Common Tasks

- **Adding an MCP tool**: `mempalace/mcp_server.py` — add handler function + TOOLS dict entry
- **Changing search**: `mempalace/searcher.py`
- **Modifying mining**: `mempalace/miner.py` (project files) or `mempalace/convo_miner.py` (transcripts)
- **Adding a storage backend**: subclass `mempalace/backends/base.py`, register in `backends/__init__.py`
- **Input validation**: `mempalace/config.py` — `sanitize_name()` / `sanitize_content()`
- **Tests**: mirror source structure in `tests/test_<module>.py`

## Two-machine sync

`./sync.sh` (mac) or `.\sync.ps1` (PC) runs `scripts/palace_sync.py`, which syncs
both directions from whichever side you are on. **`PALACE_SYNC_NOTES.md` is the
protocol** — read it before touching the sync, the mirrors, or wing naming.

Two invariants it exists to protect. `drawer_id` hashes the **full absolute**
`source_file`, so a mirror directory that moves refiles the entire corpus as
duplicate drawers; the mirrors are permanent addresses. And the sync must ship
exactly what `mempalace mine --mode convos` ingests — `CONVO_EXTENSIONS` minus
`CONVO_SKIP_DIRS` — because the Stop and SessionEnd hooks already mine the whole
project tree on the machine of origin. Filtering more than the miner leaves the
mirror holding less than the palace it mirrors; filtering less files content the
origin never did. Subagent transcripts are kept deliberately; `tool-results/` is
skipped deliberately. Change one side of that pair and you must change the other.

`scripts/palace_sync.config.json` is gitignored and describes one machine pair;
copy `palace_sync.config.example.json` to create it.

## Palace Repair — salvage first, `repair --yes` last

When the MCP server dies on every call (SIGSEGV / "Connection closed") and the
palace needs rebuilding, **`tmp/SALVAGE_RUNBOOK.md` is the first protocol to try.**
It supersedes `mempalace repair --yes` and every earlier rebuild procedure,
including the one in `tmp/PALACE_RUNBOOK.md`.

An HNSW segment that segfaults on open has a corrupt **link graph**; its vectors
are intact. `repair --yes` discards them and re-runs ONNX inference on every
chunk — that is the entire cost. Reading the vectors back out of the quarantined
`data_level0.bin` and upserting them via `ChromaCollection.upsert(embeddings=...)`
produces an identical index without embedding anything.

Measured on the live 199,152-drawer palace, 2026-07-22: **4m20s (~1,000/s) versus
~66min (~50/s)** — 16x, with salvaged vectors matching fresh embeddings at cosine
1.00000. Fall back to `repair --yes` only when the runbook's two validation gates
(unit-norm check, cosine-vs-fresh spot check) fail.

Two facts that cost real time and belong here rather than only in the runbook:
`repair-status` reports **OK while the palace is fatally corrupt**, because counts
stay within flush-lag tolerance while the graph is destroyed — the only reliable
probe is opening the collection in a throwaway process and checking for exit code
139. And `repair --yes` **empties sqlite before refilling it**, so mid-run the only
complete copy of the verbatim text is that process's RAM plus `palace.backup`;
verify the backup's row count before killing a running repair.
