#!/bin/bash
# MemPalace SessionEnd Hook — thin wrapper calling Python CLI
# All logic lives in mempalace.hooks_cli for cross-harness extensibility.
# Fires on /clear, logout, prompt-input-exit, etc. — guarantees a
# discoverable diary stub even when the AI never gets a chance to save.
INPUT=$(cat)
echo "$INPUT" | python3 -m mempalace hook run --hook session-end --harness claude-code
