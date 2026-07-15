"""Tests for the cross-process palace write lock and its backend wiring.

The lock exists because ChromaDB's ``PersistentClient`` permanently
corrupts HNSW under concurrent multi-process writes (chroma#1584;
the 2026-07-13 incident). Every mutation must serialize through it.
"""

import multiprocessing
import time

import pytest

from mempalace.palace_lock import (
    PalaceWriteLockTimeout,
    _lock_path,
    palace_write_lock,
)


def test_lock_acquires_and_releases(tmp_path):
    with palace_write_lock(str(tmp_path)):
        pass
    # Re-acquirable immediately after release.
    with palace_write_lock(str(tmp_path), timeout=1):
        pass


def test_lock_is_reentrant_within_thread(tmp_path):
    with palace_write_lock(str(tmp_path)):
        # Nested acquisition must not deadlock or raise.
        with palace_write_lock(str(tmp_path), timeout=1):
            pass
        # Still held after inner exit: another acquisition path (the
        # reentrancy set) should treat it as held.
        with palace_write_lock(str(tmp_path), timeout=1):
            pass


def test_lock_path_normalizes(tmp_path):
    a = _lock_path(str(tmp_path))
    b = _lock_path(str(tmp_path) + "/")
    assert a == b


def _hold_lock(palace_path, hold_seconds, acquired_evt):
    from mempalace.palace_lock import palace_write_lock

    with palace_write_lock(palace_path):
        acquired_evt.set()
        time.sleep(hold_seconds)


def test_lock_excludes_other_process(tmp_path):
    """A second process holding the lock blocks us until it releases."""
    ctx = multiprocessing.get_context("spawn")
    acquired = ctx.Event()
    proc = ctx.Process(target=_hold_lock, args=(str(tmp_path), 1.0, acquired))
    proc.start()
    try:
        assert acquired.wait(timeout=30), "child never acquired the lock"
        start = time.monotonic()
        with palace_write_lock(str(tmp_path), timeout=30):
            waited = time.monotonic() - start
        # We must have waited for the child's hold window (allow scheduling slack).
        assert waited > 0.3, f"lock did not exclude concurrent process (waited {waited:.2f}s)"
    finally:
        proc.join(timeout=30)


def test_lock_timeout_raises(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    acquired = ctx.Event()
    proc = ctx.Process(target=_hold_lock, args=(str(tmp_path), 3.0, acquired))
    proc.start()
    try:
        assert acquired.wait(timeout=30), "child never acquired the lock"
        with pytest.raises(PalaceWriteLockTimeout):
            with palace_write_lock(str(tmp_path), timeout=0.3):
                pass
    finally:
        proc.join(timeout=30)


def test_lock_released_when_holder_dies(tmp_path):
    """flock is released by the OS on process death — no stale locks."""
    ctx = multiprocessing.get_context("spawn")
    acquired = ctx.Event()
    proc = ctx.Process(target=_hold_lock, args=(str(tmp_path), 60.0, acquired))
    proc.start()
    try:
        assert acquired.wait(timeout=30), "child never acquired the lock"
        proc.terminate()
        proc.join(timeout=30)
        with palace_write_lock(str(tmp_path), timeout=5):
            pass
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=10)


# ── Backend wiring ────────────────────────────────────────────────────


class _RecordingLock:
    def __init__(self):
        self.entered = 0

    def __call__(self, palace_path, timeout=300.0):
        import contextlib

        outer = self

        @contextlib.contextmanager
        def _cm():
            outer.entered += 1
            yield

        return _cm()


@pytest.fixture
def recording_lock(monkeypatch):
    rec = _RecordingLock()
    monkeypatch.setattr("mempalace.backends.chroma.palace_write_lock", rec)
    return rec


def _collection_with_lock(palace_path):
    from unittest.mock import MagicMock

    from mempalace.backends.chroma import ChromaCollection

    raw = MagicMock()
    raw._client._mempalace_gen = ""  # matches the no-gen-file state of a fresh palace
    return ChromaCollection(raw, palace_path=palace_path), None


def test_collection_writes_take_lock(recording_lock, tmp_path):
    col, _ = _collection_with_lock(str(tmp_path))
    col.upsert(documents=["d"], ids=["i"])
    col.add(documents=["d2"], ids=["i2"])
    col.update(ids=["i"], metadatas=[{"k": "v"}])
    col.delete(ids=["i"])
    assert recording_lock.entered == 4


def test_collection_reads_do_not_take_lock(recording_lock, tmp_path):
    col, _ = _collection_with_lock(str(tmp_path))
    col.count()
    assert recording_lock.entered == 0


def test_collection_without_path_skips_lock(recording_lock):
    from unittest.mock import MagicMock

    from mempalace.backends.chroma import ChromaCollection

    col = ChromaCollection(MagicMock())
    col.upsert(documents=["d"], ids=["i"])
    assert recording_lock.entered == 0


# ── Cross-process write coherence (generation stamp) ─────────────────


def test_write_through_stale_system_refreshes(tmp_path, monkeypatch):
    """A write generation bumped by another process forces a true client
    rebuild (SharedSystemClient cache clear) before the next write; writes
    with no external bump do not rebuild."""
    import mempalace.backends.chroma as mod
    from mempalace.backends.chroma import ChromaBackend, _bump_generation

    backend = ChromaBackend()
    palace = str(tmp_path)
    col = backend.get_collection(palace, "mempalace_drawers", create=True)
    col.upsert(ids=["a"], documents=["a"], embeddings=[[0.1] * 4])

    clears = []
    orig = mod.SharedSystemClient.clear_system_cache
    monkeypatch.setattr(
        mod.SharedSystemClient,
        "clear_system_cache",
        staticmethod(lambda: (clears.append(1), orig())[1]),
    )

    _bump_generation(palace)  # simulate another process's write
    col.upsert(ids=["b"], documents=["b"], embeddings=[[0.2] * 4])
    assert len(clears) == 1

    col.upsert(ids=["c"], documents=["c"], embeddings=[[0.3] * 4])
    assert len(clears) == 1  # our own write kept the tag current

    assert col.count() == 3
