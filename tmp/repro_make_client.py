"""Staged repro of mempalace's own client-open path.

Runs the same steps ChromaBackend.make_client performs, individually,
so we can see which one wedges. faulthandler force-exits after 60s.
"""

import faulthandler

faulthandler.enable()
faulthandler.dump_traceback_later(60, exit=True)

PALACE = "/Users/peterwang/.mempalace/palace"

print("stage1: import mempalace chroma backend", flush=True)
from mempalace.backends import chroma as cb

print("stage1 ok", flush=True)

print("stage2: _fix_blob_seq_ids", flush=True)
cb._fix_blob_seq_ids(PALACE)
print("stage2 ok", flush=True)

print("stage3: quarantine_stale_hnsw", flush=True)
cb.quarantine_stale_hnsw(PALACE)
print("stage3 ok", flush=True)

print("stage4: PersistentClient open", flush=True)
import chromadb

client = chromadb.PersistentClient(path=PALACE)
print("stage4 ok", flush=True)

print("stage5: _maybe_auto_drain_unflushed_queue", flush=True)
cb._maybe_auto_drain_unflushed_queue(PALACE)
print("stage5 ok", flush=True)

print("ALL STAGES PASSED", flush=True)
