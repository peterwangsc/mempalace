"""Stage HNSW vector query against each collection. Watchdog 60s."""

import faulthandler

faulthandler.enable()
faulthandler.dump_traceback_later(60, exit=True)

PALACE = "/Users/peterwang/.mempalace/palace"

import chromadb

client = chromadb.PersistentClient(path=PALACE)
for col in client.list_collections():
    c = client.get_collection(col.name)
    n = c.count()
    print(f"collection {col.name}: count={n}", flush=True)
    peek = c.peek(1)
    dim = len(peek["embeddings"][0])
    print(f"  dim={dim}, hnsw cfg={c.configuration_json.get('hnsw')}", flush=True)
    print(f"  querying...", flush=True)
    r = c.query(query_embeddings=[[0.1] * dim], n_results=3, include=[])
    print(f"  query ok: {r['ids']}", flush=True)

print("ALL QUERIES PASSED", flush=True)
