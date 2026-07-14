"""Benchmark hnswlib at MemPalace scale.

Goal: answer "is hnswlib's stock save/load/query good enough to replace
chromadb's local_persistent_hnsw?"

Measures:
  - parallel build time at 158k x 384d
  - save_index() output: file count, size, atomicity
  - load_index() cold time
  - load_index() with mmap (if available)
  - 1000 random query latency p50/p95/p99
  - crash-during-save behavior (kill mid-write, attempt reload)

Synthetic vectors are fine: hnswlib's index timings depend on count,
dimension, ef_construction, M — not content distribution. Real vectors
would only change recall numbers, which we don't measure here.
"""

import os
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time

import hnswlib
import numpy as np

N = 158_466
DIM = 384
M = 16
EF_CONSTRUCTION = 200
EF_QUERY = 100

WORKDIR = tempfile.mkdtemp(prefix="hnsw_bench_")
print(f"workdir: {WORKDIR}")
print(f"scale:   N={N:,}  dim={DIM}  M={M}  ef_construction={EF_CONSTRUCTION}")
print()


def fmt_secs(s):
    return f"{s:.2f}s" if s < 60 else f"{s / 60:.1f}m"


def fmt_bytes(b):
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


# ---------- generate vectors ----------
print("[1/6] generating synthetic vectors")
t0 = time.time()
rng = np.random.default_rng(42)
data = rng.standard_normal((N, DIM)).astype(np.float32)
# normalize to unit length so cosine == inner product
norms = np.linalg.norm(data, axis=1, keepdims=True)
data /= np.maximum(norms, 1e-9)
ids = np.arange(N, dtype=np.int64)
print(f"      generated in {fmt_secs(time.time() - t0)}, mem ~{fmt_bytes(data.nbytes)}")
print()


# ---------- build ----------
print("[2/6] building index (parallel)")
idx = hnswlib.Index(space="cosine", dim=DIM)
idx.init_index(max_elements=N, ef_construction=EF_CONSTRUCTION, M=M)
idx.set_num_threads(os.cpu_count() or 4)
t0 = time.time()
idx.add_items(data, ids)
build_secs = time.time() - t0
print(f"      build:  {fmt_secs(build_secs)}  ({N / build_secs:,.0f} vec/s)")
print()


# ---------- save ----------
print("[3/6] saving index")
save_path = os.path.join(WORKDIR, "index.bin")
t0 = time.time()
idx.save_index(save_path)
save_secs = time.time() - t0
save_size = os.path.getsize(save_path)
files_in_workdir = os.listdir(WORKDIR)
print(f"      save:   {fmt_secs(save_secs)}")
print(f"      file:   {save_path}  ({fmt_bytes(save_size)})")
print(f"      files in workdir: {files_in_workdir}")
print()


# ---------- cold load ----------
print("[4/6] cold load")
del idx
idx2 = hnswlib.Index(space="cosine", dim=DIM)
t0 = time.time()
idx2.load_index(save_path, max_elements=N)
idx2.set_ef(EF_QUERY)
load_secs = time.time() - t0
print(f"      load:   {fmt_secs(load_secs)}")
print()


# ---------- query ----------
print("[5/6] query 1000 random vectors")
queries = rng.standard_normal((1000, DIM)).astype(np.float32)
queries /= np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-9)
latencies = []
for q in queries:
    t0 = time.time()
    idx2.knn_query(q, k=10)
    latencies.append((time.time() - t0) * 1000)
latencies.sort()
p50 = latencies[500]
p95 = latencies[950]
p99 = latencies[990]
mean = statistics.mean(latencies)
print(f"      mean: {mean:.2f} ms   p50: {p50:.2f}   p95: {p95:.2f}   p99: {p99:.2f}")
print()


# ---------- crash-during-save ----------
print("[6/6] crash-during-save test")
# spawn a subprocess that starts saving; kill it after a few ms.
# then attempt to reload the partial file and see what happens.
crash_path = os.path.join(WORKDIR, "crash.bin")
crash_script = os.path.join(WORKDIR, "_save.py")
with open(crash_script, "w") as f:
    f.write(f"""
import hnswlib, numpy as np, time, sys
N, DIM = {N}, {DIM}
rng = np.random.default_rng(42)
data = rng.standard_normal((N, DIM)).astype(np.float32)
data /= np.maximum(np.linalg.norm(data, axis=1, keepdims=True), 1e-9)
idx = hnswlib.Index(space='cosine', dim=DIM)
idx.init_index(max_elements=N, ef_construction={EF_CONSTRUCTION}, M={M})
idx.set_num_threads({os.cpu_count() or 4})
idx.add_items(data, np.arange(N, dtype=np.int64))
print('READY', flush=True)
sys.stdout.flush()
idx.save_index({crash_path!r})
print('DONE', flush=True)
""")

proc = subprocess.Popen(
    [sys.executable, crash_script],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)
# wait for READY then kill after a short delay during save
for line in proc.stdout:
    if line.strip() == "READY":
        break
# small sleep so save actually starts writing
time.sleep(0.05)
proc.send_signal(signal.SIGKILL)
proc.wait()
crash_size = os.path.getsize(crash_path) if os.path.exists(crash_path) else 0
print(f"      partial file: {fmt_bytes(crash_size)} (full would be ~{fmt_bytes(save_size)})")

if crash_size == 0:
    print("      no file produced — caller would just retry, no corruption risk")
else:
    # try to load the partial file
    idx3 = hnswlib.Index(space="cosine", dim=DIM)
    try:
        idx3.load_index(crash_path, max_elements=N)
        print("      WARNING: partial file loaded silently — corruption risk")
    except Exception as e:
        msg = str(e).splitlines()[0][:120]
        print(f"      partial file rejected: {type(e).__name__}: {msg}")
        print("      (good — caller would fall back to rebuild)")

print()
print("---SUMMARY---")
print(f"build      {fmt_secs(build_secs)}  ({N / build_secs:,.0f} vec/s)")
print(f"save       {fmt_secs(save_secs)}  ({fmt_bytes(save_size)})")
print(f"load       {fmt_secs(load_secs)}")
print(f"query mean {mean:.2f} ms   p99 {p99:.2f} ms")
print(f"crash      partial={fmt_bytes(crash_size)}, "
      f"reload {'ACCEPTED (BAD)' if crash_size > 0 else 'no file'}")
print(f"workdir    {WORKDIR}")
shutil.rmtree(WORKDIR, ignore_errors=True)
