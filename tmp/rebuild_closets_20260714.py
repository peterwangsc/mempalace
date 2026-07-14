#!/usr/bin/env python3.13
"""One-off: rebuild the mempalace_closets HNSW index after the 2026-07-14 incident.

Mirrors repair.rebuild_index (which is hardcoded to mempalace_drawers):
extract all closets via get() (documents live in the sqlite METADATA
segment, unaffected by the emptied vector index), delete + recreate the
collection with the small-collection bloat guard, upsert back so the
embedding function re-embeds. ~6.8k entries, a couple of minutes.

Run:  python3.13 tmp/rebuild_closets_20260714.py
"""

import sqlite3

from mempalace.backends.chroma import ChromaBackend, _HNSW_BLOAT_GUARD_SMALL, hnsw_capacity_status

PALACE = "/Users/peterwang/.mempalace/palace"
COLLECTION = "mempalace_closets"
BATCH = 1000

backend = ChromaBackend()
col = backend.get_collection(PALACE, COLLECTION)
total = col.count()
print(f"closets reported by chroma: {total}")

conn = sqlite3.connect(f"file:{PALACE}/chroma.sqlite3?mode=ro", uri=True)
sqlite_truth = conn.execute(
    """SELECT COUNT(*) FROM embeddings e
       JOIN segments s ON e.segment_id = s.id
       JOIN collections c ON s.collection = c.id
       WHERE c.name = ? AND s.scope = 'METADATA'""",
    (COLLECTION,),
).fetchone()[0]
conn.close()
print(f"closets in sqlite ground truth: {sqlite_truth}")

ids, docs, metas = [], [], []
offset = 0
while True:
    batch = col.get(limit=BATCH, offset=offset, include=["documents", "metadatas"])
    if not batch["ids"]:
        break
    ids.extend(batch["ids"])
    docs.extend(batch["documents"])
    metas.extend(batch["metadatas"])
    offset += len(batch["ids"])
print(f"extracted: {len(ids)}")

if len(ids) < sqlite_truth:
    raise SystemExit(
        f"ABORT: extracted {len(ids)} < sqlite ground truth {sqlite_truth} — "
        "refusing to delete the collection on a short extraction."
    )

print("deleting + recreating collection...")
backend.delete_collection(PALACE, COLLECTION)
new_col = backend.create_collection(PALACE, COLLECTION, metadata_overrides=_HNSW_BLOAT_GUARD_SMALL)

filed = 0
for i in range(0, len(ids), BATCH):
    new_col.upsert(ids=ids[i : i + BATCH], documents=docs[i : i + BATCH], metadatas=metas[i : i + BATCH])
    filed += len(ids[i : i + BATCH])
    print(f"re-filed {filed}/{len(ids)}")

cap = hnsw_capacity_status(PALACE, COLLECTION)
print(f"post-rebuild: {cap['status']} hnsw={cap['hnsw_count']} sqlite={cap['sqlite_count']}")
