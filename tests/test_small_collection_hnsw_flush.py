"""Regression tests for the ``hnsw:sync_threshold=50_000`` flush gap.

Hypothesis under test
---------------------
PR #1191 set::

    _HNSW_BLOAT_GUARD = {"hnsw:batch_size": 50_000, "hnsw:sync_threshold": 50_000}

at every collection-creation site (``backends/chroma.py``,
``mcp_server.py``). The empirical validation in that PR was on a
39,792-drawer rebuild — i.e. a single large collection that crosses
50,000 items.

Collections that legitimately stay small (notably ``mempalace_closets``,
which holds ~6,000 entries on a ~110k-drawer palace) never reach the
sync threshold. chromadb 1.5.x persists ``index_metadata.pickle`` only
when ``sync_threshold`` is hit, so the small-collection HNSW segment is
written to disk with **no flushed metadata** and ``hnsw_capacity_status``
reports it as ``DIVERGED, never flushed``.

Symptoms in production (Apr 2026):
    * ``mempalace repair-status`` reports ``mempalace_closets`` DIVERGED
      with ``hnsw_count = None`` despite N drawers in sqlite.
    * Vector search over closets returns empty.
    * ``mempalace repair --mode legacy`` SIGSEGVs trying to rebuild the
      missing-pickle segment.

These tests reproduce the gap on a fresh tmp palace, so a fix can be
verified without touching the user's real palace.

Each test creates a small collection, upserts < 50_000 items, drops
client references, gc.collects to trigger chromadb's ``__del__`` flush
path (the only flush opportunity in the small-collection regime), then
reopens with a fresh client and probes with the public capacity helper.

If the hypothesis is right, these tests fail today.
"""

from __future__ import annotations

import gc
import os

import chromadb
import pytest

from mempalace.backends.chroma import (
    ChromaBackend,
    _hnsw_element_count,
    _vector_segment_id,
    hnsw_capacity_status,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _drop_chroma_handles() -> None:
    """Drop chromadb's internal singleton and force a gc cycle.

    chromadb caches a ``System`` per palace path; without resetting it,
    a subsequent ``PersistentClient(path=...)`` call returns the same
    in-memory state and any flush-on-teardown path is bypassed.
    """
    try:
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception:  # pragma: no cover — defensive
        pass
    gc.collect()


def _upsert_small(collection, n: int) -> None:
    """Upsert ``n`` trivially-distinct documents into ``collection``.

    Passes pre-computed dummy embeddings so we skip the ONNX embedder
    entirely — keeps the parametrized 3000-item test under a second
    while still exercising the same chromadb persist path that production
    closet upserts hit.
    """
    # 384 matches the default MiniLM dim. We seed each row with a
    # slightly-different unit-ish vector so cosine distance stays well-
    # defined; the actual values don't matter for HNSW persistence
    # behavior, only that *some* embedding is supplied.
    def _vec(i: int) -> list[float]:
        v = [0.0] * 384
        v[i % 384] = 1.0
        v[(i + 17) % 384] = 0.5
        return v

    collection.upsert(
        ids=[f"d-{i}" for i in range(n)],
        documents=[f"doc {i}" for i in range(n)],
        embeddings=[_vec(i) for i in range(n)],
        metadatas=[{"wing": "test", "room": "small", "idx": i} for i in range(n)],
    )


# ── Core regression tests ────────────────────────────────────────────


@pytest.mark.parametrize("n_items", [100, 1000, 3000])
def test_small_collection_persists_hnsw_pickle_after_graceful_close(tmp_path, n_items):
    """A small collection (N << sync_threshold) must persist its HNSW
    segment metadata on graceful client teardown.

    Without persistence, ``hnsw_capacity_status`` returns
    ``hnsw_count=None`` and reports ``DIVERGED, never flushed`` — the
    exact production symptom on ``mempalace_closets``.
    """
    palace = str(tmp_path / "palace")

    backend = ChromaBackend()
    col = backend.get_collection(palace, collection_name="mempalace_closets", create=True)
    _upsert_small(col, n_items)

    # Force chromadb to flush whatever it intends to flush on teardown.
    del col
    del backend
    _drop_chroma_handles()

    seg_id = _vector_segment_id(palace, "mempalace_closets")
    assert seg_id is not None, "VECTOR segment row must exist after upsert"

    hnsw_count = _hnsw_element_count(palace, seg_id)
    assert hnsw_count is not None, (
        f"index_metadata.pickle missing after upserting {n_items} items and closing client. "
        f"sync_threshold=50_000 prevents flush for collections that stay small. "
        f"This is the production symptom on mempalace_closets."
    )
    assert hnsw_count == n_items, (
        f"HNSW persisted {hnsw_count} elements but expected {n_items} after small upsert."
    )


def test_small_closet_collection_query_returns_results_after_reopen(tmp_path):
    """End-to-end: small closet collection survives client teardown +
    reopen and answers a vector query.

    The closet pipeline upserts documents one source at a time, then the
    process ends. If teardown does not flush, the next process opens a
    palace whose closet HNSW returns nothing for any query.
    """
    palace = str(tmp_path / "palace")

    # First "session" — populate.
    backend = ChromaBackend()
    col = backend.get_collection(palace, collection_name="mempalace_closets", create=True)
    _upsert_small(col, 50)
    del col
    del backend
    _drop_chroma_handles()

    # Second "session" — fresh client, query.
    client2 = chromadb.PersistentClient(path=palace)
    col2 = client2.get_collection("mempalace_closets")
    assert col2.count() == 50, "sqlite-side count must reflect upserted items"

    result = col2.query(query_texts=["topic 3"], n_results=5)
    ids = result["ids"][0] if result["ids"] else []
    assert ids, (
        "Vector query returned zero results from a populated small collection — "
        "HNSW segment never flushed metadata, so chromadb has nothing to search. "
        "Closet vector search is silently dead in this configuration."
    )


def test_repair_status_does_not_flag_healthy_small_collection_as_diverged(tmp_path):
    """``hnsw_capacity_status`` must not report DIVERGED for a small
    collection that has been upserted and gracefully closed.

    Without a flush, ``hnsw_count`` is None and the status helper raises
    the ``never flushed`` flag — false positive for a healthy palace.
    """
    palace = str(tmp_path / "palace")

    # Use 3000 items — above the _HNSW_DIVERGENCE_ABSOLUTE=2000 threshold,
    # so a missing pickle would actively be flagged DIVERGED. This is the
    # exact prod symptom on a 6,054-item closet collection: capacity probe
    # reports "DIVERGED, never flushed" and recommends `mempalace repair`.
    backend = ChromaBackend()
    col = backend.get_collection(palace, collection_name="mempalace_closets", create=True)
    _upsert_small(col, 3000)
    del col
    del backend
    _drop_chroma_handles()

    info = hnsw_capacity_status(palace, "mempalace_closets")
    assert info["sqlite_count"] == 3000
    assert info["diverged"] is False, (
        f"hnsw_capacity_status reported DIVERGED on a healthy 200-item closet "
        f"collection: {info!r}. The sync_threshold=50_000 guard hides the "
        f"flush from any palace whose closet count never crosses that line."
    )


# ── Cause-pinning control test ────────────────────────────────────────


def test_low_sync_threshold_does_persist_small_collection(tmp_path):
    """Control: with ``sync_threshold`` low enough to be crossed by the
    upsert, the pickle DOES get written.

    Bypasses ``ChromaBackend`` and creates the collection directly with
    ``hnsw:sync_threshold=10`` so we can isolate the cause to the
    50_000 setting rather than to chromadb's overall flush behavior or
    the test fixture environment. If this test passes while the ones
    above fail, the bug is unambiguously the ``_HNSW_BLOAT_GUARD``
    threshold being too high for small collections.
    """
    palace = str(tmp_path / "palace")
    os.makedirs(palace, exist_ok=True)

    client = chromadb.PersistentClient(path=palace)
    col = client.get_or_create_collection(
        "mempalace_closets",
        metadata={
            "hnsw:space": "cosine",
            "hnsw:num_threads": 1,
            "hnsw:batch_size": 10,
            "hnsw:sync_threshold": 10,
        },
    )
    _upsert_small(col, 200)

    del col
    del client
    _drop_chroma_handles()

    seg_id = _vector_segment_id(palace, "mempalace_closets")
    assert seg_id is not None
    hnsw_count = _hnsw_element_count(palace, seg_id)
    assert hnsw_count is not None and hnsw_count > 0, (
        "Even with sync_threshold=10 the pickle did not flush — the bug is not "
        "the threshold setting; investigate chromadb flush semantics instead."
    )


# ── Cross-check: large collection still flushes ──────────────────────


@pytest.mark.slow
def test_large_collection_still_flushes(tmp_path):
    """Counter-test for the cause hypothesis: a collection that crosses
    the 50_000 threshold must still flush as PR #1191 intended.

    Marked ``slow`` because 50_001 upserts is heavyweight; not part of
    the default suite but available via ``pytest -m slow``. Demonstrates
    the fix surface: any change must keep this passing while making the
    small-collection tests pass.
    """
    pytest.importorskip("numpy")
    palace = str(tmp_path / "palace")

    backend = ChromaBackend()
    col = backend.get_collection(palace, collection_name="mempalace_drawers", create=True)
    # Cross sync_threshold=50_000.
    n = 50_500
    col.upsert(
        ids=[f"d-{i}" for i in range(n)],
        documents=[f"doc {i}" for i in range(n)],
    )
    del col
    del backend
    _drop_chroma_handles()

    seg_id = _vector_segment_id(palace, "mempalace_drawers")
    assert seg_id is not None
    hnsw_count = _hnsw_element_count(palace, seg_id)
    assert hnsw_count is not None and hnsw_count >= 50_000
