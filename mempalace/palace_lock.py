"""Cross-process palace write lock.

ChromaDB's ``PersistentClient`` is not safe for concurrent writes from
multiple processes (chroma-core/chroma#1584 — permanent HNSW corruption,
reproduced on 1.5.7). MemPalace routinely has several writer processes
alive at once: each Claude Code session's MCP server, hook-spawned
``mempalace mine --cursor`` subprocesses on Stop/PreCompact/SessionEnd,
and CLI commands. The 2026-07-13 incident (four SIGSEGVs, full index
rebuild) traced to exactly this: a compact's miner flushing HNSW while
other processes held live clients on the same palace.

``palace_write_lock`` serializes every chroma mutation across processes:

* **Exclusive, blocking with timeout** — writers queue instead of
  interleaving. Chroma flushes HNSW synchronously inside write calls, so
  serializing the calls serializes the flushes.
* **Reentrant per thread** — the client-open healing gate holds the lock
  while ``recover_unflushed_buffer`` runs, which itself writes through
  the locked collection wrapper.
* **Crash-safe** — ``flock``/``msvcrt`` locks are released by the OS
  when the holder dies; no stale-lock cleanup or PID files needed.

Lock ordering: callers that also take the per-file ``mine_lock`` or the
per-palace ``mine_palace_lock`` acquire those FIRST and this lock last
(it is the innermost, shortest-held lock). Never take a mine lock while
holding this one.
"""

import contextlib
import hashlib
import os
import threading
import time

_LOCK_DIR = os.path.join(os.path.expanduser("~"), ".mempalace", "locks")
_POLL_INTERVAL = 0.05

_local = threading.local()


class PalaceWriteLockTimeout(TimeoutError):
    """Raised when the palace write lock cannot be acquired within the timeout."""


def _lock_path(palace_path: str) -> str:
    resolved = os.path.normcase(os.path.realpath(os.path.expanduser(palace_path)))
    key = hashlib.sha256(resolved.encode()).hexdigest()[:16]
    return os.path.join(_LOCK_DIR, f"palace_write_{key}.lock")


def _try_lock(lf) -> bool:
    if os.name == "nt":
        import msvcrt

        try:
            msvcrt.locking(lf.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    else:
        import fcntl

        try:
            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False


def _unlock(lf) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lf.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lf, fcntl.LOCK_UN)
    except Exception:
        pass


@contextlib.contextmanager
def palace_write_lock(palace_path: str, timeout: float = 300.0):
    """Hold the exclusive cross-process write lock for ``palace_path``.

    Blocks up to ``timeout`` seconds waiting for other writers, then
    raises :class:`PalaceWriteLockTimeout`. Reentrant within a thread:
    nested acquisitions of the same palace's lock are no-ops, so a
    lock-holding caller can safely invoke helpers that also lock.

    The generous default timeout reflects the worst legitimate hold: a
    miner embedding one upsert batch. Waiting beats interleaving — an
    interleaved write corrupts the index permanently.
    """
    lock_path = _lock_path(palace_path)

    held: set = getattr(_local, "held", None) or set()
    _local.held = held
    if lock_path in held:
        yield
        return

    os.makedirs(_LOCK_DIR, exist_ok=True)
    lf = open(lock_path, "w")
    deadline = time.monotonic() + timeout
    try:
        while not _try_lock(lf):
            if time.monotonic() >= deadline:
                raise PalaceWriteLockTimeout(
                    f"could not acquire palace write lock for {palace_path} "
                    f"within {timeout:.0f}s — another process is writing"
                )
            time.sleep(_POLL_INTERVAL)
        held.add(lock_path)
        try:
            yield
        finally:
            held.discard(lock_path)
            _unlock(lf)
    finally:
        lf.close()
