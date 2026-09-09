# Codex conversation hooks on personal-v3

`scripts/codex_memory_hook.py` bridges current Codex history to the existing
incremental miner. Stop and SessionEnd both pass their JSON stdin to this script.
It immediately detaches a worker; export and mining run under a shared local lock.
SessionEnd's three-second limit therefore does not truncate a mine.

The worker reads `CODEX_HOME/thread_history_1.sqlite` read-only, selecting authored
`userMessage` and `agentMessage` items. It falls back to the supplied rollout path
for older sessions, including both legacy message events and `item_completed`.
Reasoning and synthetic `response_item` context are excluded. Subagent sessions
are accepted when the harness emits their hooks, without scanning unrelated history.

Stable exports live at
`~/.codex/mempalace-transcripts/<origin-project-wing>/<session-id>/transcript.jsonl`.
Do not move them: source paths participate in drawer IDs. Original items accompany
the authored text. Repeated events append only new records; a changed saved prefix
raises an error and preserves the existing file. The cursor miner retries the same
stable file safely. Errors remain in `~/.codex/mempalace-hooks.log`; failed request
files remain in `~/.codex/mempalace-hook-inputs/` for manual replay with `--worker`.
Successful requests are removed. No automatic cross-machine mirror was added.

Configure user `~/.codex/hooks.json` with Stop and SessionEnd command handlers
pointing to the venv Python and this script using absolute paths, with timeout 3.
Keep any unrelated hook handlers. On Windows the command can use forward slashes:

```json
{"hooks":{"Stop":[{"hooks":[{"type":"command","command":"C:/Users/pewa/code/mempalace/.venv/Scripts/python.exe -X utf8 C:/Users/pewa/code/mempalace/scripts/codex_memory_hook.py","timeout":3}]}],"SessionEnd":[{"hooks":[{"type":"command","command":"C:/Users/pewa/code/mempalace/.venv/Scripts/python.exe -X utf8 C:/Users/pewa/code/mempalace/scripts/codex_memory_hook.py","timeout":3}]}]}}
```

Restart Codex and review/trust both definitions with `/hooks`. Full-access shell
permissions do not replace Codex's separate hook trust requirement. See
[OpenAI's hook documentation](https://learn.chatgpt.com/docs/hooks).

## Upstream review, September 9, 2026

Reviewed upstream default `develop` at `f9297a228586ebbfc5ac37763793d6c89ef69104`
against personal-v3 `63c0401`. Upstream includes substantial backend and hook
changes, but its Codex parser still only recognizes legacy message events. A trial
merge produced 17 conflicted files, including personal cursor mining, canonical
wing routing, generation locks and HNSW repair. The trial was aborted; no upstream
merge is claimed. This focused patch preserves the existing live palace semantics
while adding current Codex support. Broad upstream integration remains separate work.
