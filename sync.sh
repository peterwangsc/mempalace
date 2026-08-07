#!/usr/bin/env bash
# Sync this palace with its peer, both directions. Run from the repo root:
#   ./sync.sh              pull then push
#   ./sync.sh --dry-run    show the delta, ship nothing
#   ./sync.sh --pull-only
set -euo pipefail
cd "$(dirname "$0")"
# chroma reopens its client mid-mine and macOS defaults to 256 descriptors
ulimit -n 10240 2>/dev/null || true
exec ./.venv/bin/python scripts/palace_sync.py "$@"
