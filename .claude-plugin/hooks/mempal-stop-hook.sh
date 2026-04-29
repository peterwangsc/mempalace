#!/bin/bash
# MemPalace Stop Hook — DISABLED 2026-04-28
# Auto-ingest mine subprocess SIGSEGVs in chromadb_rust_bindings HNSW
# write path on this palace state (predates today; first observed
# Apr 14, compounded by Apr 20 closet retrofit + OSS update). Original
# wrapper preserved below for git-revert when chromadb is fixed or the
# palace is rebuilt from scratch on a working version.
#
# INPUT=$(cat)
# echo "$INPUT" | python3 -m mempalace hook run --hook stop --harness claude-code
exit 0
