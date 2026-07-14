"""Staged repro: does a bare Chroma open against the real palace wedge?

Each stage prints before it runs (flushed). faulthandler dumps all thread
stacks and force-exits after 45s so this can never hang the caller.
"""

import faulthandler
import sys

faulthandler.enable()
faulthandler.dump_traceback_later(45, exit=True)

PALACE = "/Users/peterwang/.mempalace/palace"

print("stage1: import chromadb", flush=True)
import chromadb

print(f"stage1 ok (chromadb {chromadb.__version__})", flush=True)

print("stage2: PersistentClient open", flush=True)
client = chromadb.PersistentClient(path=PALACE)
print("stage2 ok", flush=True)

print("stage3: list_collections", flush=True)
cols = client.list_collections()
print(f"stage3 ok: {len(cols)} collections", flush=True)

name = cols[0].name if cols else None
print(f"stage4: get_collection({name!r}) + count", flush=True)
col = client.get_collection(name)
print(f"stage4 ok: count={col.count()}", flush=True)

print("stage5: get(limit=1)", flush=True)
r = col.get(limit=1, include=[])
print(f"stage5 ok: ids={r['ids']}", flush=True)

print("ALL STAGES PASSED", flush=True)
sys.exit(0)
