"""Regression tests for the HNSW-segment quarantine defenses.

Two real-world incidents motivated these tests, both observed on a
~110k-drawer chromadb-1.5.7 palace in 2026-04:

1. **Mid-write corruption masked as "fresh".** ``_segment_appears_healthy``
   treats a missing ``index_metadata.pickle`` as "segment hasn't flushed
   yet, healthy". That's true for a brand-new empty segment (where
   ``data_level0.bin`` is 0 bytes or absent) but wrong for a segment
   with populated HNSW data files and no pickle — that's a chromadb
   crash mid-flush or a sync_threshold-suppressed flush, exactly the
   pathology the closet-flush incident produced. ``quarantine_stale_hnsw``
   then refuses to rename the corrupt segment because it "looks healthy",
   leaving it in place to SIGSEGV the next ``PersistentClient(...)`` open.

2. **``_client`` skips quarantine that ``make_client`` performs.** The
   static ``ChromaBackend.make_client`` calls ``quarantine_stale_hnsw``
   once per palace per process (gated by ``_quarantined_paths``). The
   instance-method ``_client`` (used by every ``get_collection`` call —
   including the path ``mempalace repair --mode legacy`` takes) does
   not. Repair therefore opens chromadb against any stale-corrupt
   segment without protection and segfaults.

These tests pin both defenses so a future refactor doesn't reintroduce
the gap.
"""

from __future__ import annotations

import os
import time
from pathlib import Path


from mempalace.backends.chroma import (
    ChromaBackend,
    _segment_appears_healthy,
    quarantine_stale_hnsw,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _write_segment(
    seg_dir: Path,
    *,
    data_level0_size: int,
    pickle_bytes: bytes | None,
    age_seconds: float = 0.0,
) -> None:
    """Construct a fake HNSW segment dir with controllable on-disk shape.

    ``data_level0_size`` bytes get written to ``data_level0.bin`` (zero
    means "empty fresh segment", non-zero means "had real HNSW data").
    ``pickle_bytes`` writes ``index_metadata.pickle`` verbatim, or skips
    the file if ``None``. ``age_seconds`` ages every file's mtime that
    many seconds into the past — used to clear the 300s freshness gate.
    """
    seg_dir.mkdir(parents=True, exist_ok=True)
    (seg_dir / "data_level0.bin").write_bytes(b"\x00" * data_level0_size)
    (seg_dir / "header.bin").write_bytes(b"\x00" * 100)
    (seg_dir / "length.bin").write_bytes(b"\x00" * 400)
    (seg_dir / "link_lists.bin").write_bytes(b"")
    if pickle_bytes is not None:
        (seg_dir / "index_metadata.pickle").write_bytes(pickle_bytes)

    if age_seconds:
        backdate = time.time() - age_seconds
        for entry in seg_dir.iterdir():
            os.utime(entry, (backdate, backdate))


def _valid_pickle_bytes() -> bytes:
    """A minimal pickle-protocol-2 byte sequence that passes the format
    sniff: starts with ``0x80`` and ends with ``0x2e`` (STOP).

    The sniff in ``_segment_appears_healthy`` only checks the first two
    bytes (proto marker) and last byte (STOP) plus a 16-byte minimum
    size. The interior bytes don't need to deserialize cleanly because
    the function never deserializes — it only checks the framing.
    """
    return b"\x80\x02" + b"\x00" * 30 + b"\x2e"


# ── _segment_appears_healthy ─────────────────────────────────────────


def test_segment_appears_healthy_rejects_mid_write_corruption(tmp_path):
    """A segment with populated ``data_level0.bin`` AND no
    ``index_metadata.pickle`` is mid-write corruption, not freshness.

    This is the exact on-disk shape the closet-flush incident produced:
    chromadb wrote HNSW data into the segment dir but never persisted
    its metadata pickle (sync_threshold=50_000 was never crossed). The
    next ``PersistentClient(...)`` open against such a segment SIGSEGVs
    in the chromadb 1.5.x Rust binding's HNSW load path.

    The sniff function must return False here so ``quarantine_stale_hnsw``
    renames the dir before chromadb opens it.
    """
    seg = tmp_path / "abc123-vector-segment"
    _write_segment(seg, data_level0_size=200_000, pickle_bytes=None)

    assert _segment_appears_healthy(str(seg)) is False, (
        "non-empty data_level0.bin + missing pickle is mid-write corruption — "
        "the closet-flush incident leaves segments in exactly this shape, "
        "and treating it as 'healthy' lets the next PersistentClient open "
        "SIGSEGV in chromadb's Rust HNSW loader."
    )


def test_segment_appears_healthy_accepts_truly_fresh_segment(tmp_path):
    """A segment with empty/absent ``data_level0.bin`` and no pickle is
    a truly fresh segment that hasn't received any writes yet. Renaming
    it would orphan nothing; sniff must return True so quarantine
    leaves it alone.
    """
    seg = tmp_path / "fresh-vector-segment"
    _write_segment(seg, data_level0_size=0, pickle_bytes=None)

    assert _segment_appears_healthy(str(seg)) is True


def test_segment_appears_healthy_accepts_flushed_segment(tmp_path):
    """A segment with a complete pickle (proto marker + STOP byte +
    minimum size) is a flushed, healthy segment regardless of
    data_level0.bin contents."""
    seg = tmp_path / "flushed-vector-segment"
    _write_segment(seg, data_level0_size=200_000, pickle_bytes=_valid_pickle_bytes())

    assert _segment_appears_healthy(str(seg)) is True


def test_segment_appears_healthy_rejects_truncated_pickle(tmp_path):
    """A pickle that's too small to be a real chromadb metadata file is
    truncation, not a healthy segment."""
    seg = tmp_path / "truncated-vector-segment"
    _write_segment(seg, data_level0_size=200_000, pickle_bytes=b"\x80\x02\x2e")  # 3 bytes < 16

    assert _segment_appears_healthy(str(seg)) is False


# ── quarantine_stale_hnsw ─────────────────────────────────────────────


def test_quarantine_renames_mid_write_corrupt_segment(tmp_path):
    """End-to-end quarantine path: a segment that's both stale (sqlite
    newer than HNSW by > 300s) AND has the mid-write-corruption shape
    must be renamed to ``<seg>.drift-<timestamp>``.

    Reproduces the closet-flush incident: chromadb wrote HNSW data but
    never the pickle, then the next process opens the palace, sqlite
    advances on first write, and the HNSW dir lags. Without this
    quarantine, the next ``PersistentClient`` open SIGSEGVs.
    """
    palace = tmp_path / "palace"
    palace.mkdir()
    sqlite = palace / "chroma.sqlite3"
    sqlite.write_bytes(b"")  # presence is enough; we don't read it

    # Corrupt segment: 200 KB data, no pickle, aged 600s into the past.
    seg = palace / "11111111-2222-3333-4444-555555555555"
    _write_segment(seg, data_level0_size=200_000, pickle_bytes=None, age_seconds=600.0)

    # sqlite is "now" (newer than the aged segment).
    os.utime(sqlite, None)

    moved = quarantine_stale_hnsw(str(palace), stale_seconds=300.0)

    assert len(moved) == 1, f"expected exactly one quarantine, got {moved!r}"
    assert ".drift-" in moved[0]
    assert not seg.exists(), "original corrupt seg dir should have been renamed"


def test_quarantine_leaves_flushed_segment_alone(tmp_path):
    """A segment that has a valid pickle is flushed and healthy even
    if its mtime trails sqlite by > 300s — chromadb 1.5.x's async
    flush makes drift the steady state. Don't quarantine."""
    palace = tmp_path / "palace"
    palace.mkdir()
    sqlite = palace / "chroma.sqlite3"
    sqlite.write_bytes(b"")

    seg = palace / "22222222-3333-4444-5555-666666666666"
    _write_segment(
        seg, data_level0_size=200_000, pickle_bytes=_valid_pickle_bytes(), age_seconds=600.0
    )
    os.utime(sqlite, None)

    moved = quarantine_stale_hnsw(str(palace), stale_seconds=300.0)
    assert moved == [], f"flushed segment should not have been quarantined: {moved!r}"
    assert seg.exists()


def test_quarantine_leaves_truly_fresh_segment_alone(tmp_path):
    """An empty fresh segment (no data, no pickle) hasn't started
    flushing yet — sqlite is naturally newer. Don't quarantine; chromadb
    will populate it on next write."""
    palace = tmp_path / "palace"
    palace.mkdir()
    sqlite = palace / "chroma.sqlite3"
    sqlite.write_bytes(b"")

    seg = palace / "33333333-4444-5555-6666-777777777777"
    _write_segment(seg, data_level0_size=0, pickle_bytes=None, age_seconds=600.0)
    os.utime(sqlite, None)

    moved = quarantine_stale_hnsw(str(palace), stale_seconds=300.0)
    assert moved == []
    assert seg.exists()


# ── ChromaBackend._client quarantine wiring ───────────────────────────


def test_get_collection_quarantines_stale_segments_on_first_open(tmp_path):
    """``ChromaBackend.get_collection`` (which goes through ``_client``)
    must invoke ``quarantine_stale_hnsw`` at least once per palace per
    process — same protection ``make_client`` provides.

    Today's incident: ``mempalace repair --mode legacy`` opens the
    palace via ``ChromaBackend().get_collection(...)`` → ``_client`` →
    ``chromadb.PersistentClient(...)``. ``_client`` skips
    ``quarantine_stale_hnsw``, so any stale-corrupt segment goes
    straight into chromadb's loader and SIGSEGVs.

    The fix is to mirror ``make_client``'s once-per-palace-per-process
    quarantine gate inside ``_client``. This test plants a corrupt
    segment in a tmp palace, calls ``get_collection``, and asserts the
    corrupt dir was renamed before chromadb saw it.
    """
    palace = tmp_path / "palace"
    palace.mkdir()
    # Seed sqlite via a real chromadb client first so the schema is valid,
    # so the eventual reopen path matches production rather than failing
    # at the schema layer for unrelated reasons.
    backend_setup = ChromaBackend()
    backend_setup.get_collection(
        str(palace), collection_name="mempalace_drawers", create=True
    )

    # Plant a corrupt VECTOR segment dir alongside the real one. The
    # corrupt dir is NOT referenced in the segments table — chromadb
    # would ignore it from a sysdb perspective — but quarantine_stale_hnsw
    # iterates the directory and renames anything that fails the
    # integrity gate, regardless of sysdb registration. Renaming an
    # unreferenced dir is a no-op for chromadb but locks down the
    # defense.
    corrupt_seg = palace / "deadbeef-dead-beef-dead-beefdeadbeef"
    _write_segment(corrupt_seg, data_level0_size=200_000, pickle_bytes=None, age_seconds=600.0)

    sqlite = palace / "chroma.sqlite3"
    os.utime(sqlite, None)

    # Reset the per-process quarantine gate so this test starts fresh
    # regardless of test ordering.
    ChromaBackend._quarantined_paths.discard(str(palace))

    # Open via ChromaBackend (the production path repair uses).
    backend = ChromaBackend()
    backend.get_collection(str(palace), collection_name="mempalace_drawers", create=False)

    # The corrupt dir should now be quarantined.
    assert not corrupt_seg.exists(), (
        "ChromaBackend.get_collection did not quarantine the stale-corrupt "
        "segment. _client needs to call quarantine_stale_hnsw with the "
        "same once-per-palace-per-process gate make_client uses, otherwise "
        "`mempalace repair --mode legacy` walks straight into the segment "
        "and SIGSEGVs in chromadb's Rust HNSW loader (the exact today's "
        "production failure mode)."
    )
    drift_dirs = [p for p in palace.iterdir() if p.name.startswith(corrupt_seg.name)]
    assert any(".drift-" in p.name for p in drift_dirs), (
        f"expected a .drift-<ts> rename of {corrupt_seg.name}, "
        f"got {[p.name for p in drift_dirs]}"
    )


def test_get_collection_quarantine_runs_only_once_per_palace_per_process(tmp_path):
    """Repeated ``get_collection`` calls against the same palace must
    not re-fire ``quarantine_stale_hnsw`` — that scan is O(segments) and
    fires on every call would thrash a daemon's open path. The
    ``_quarantined_paths`` gate protects against this.
    """
    palace = tmp_path / "palace"
    palace.mkdir()
    backend_setup = ChromaBackend()
    backend_setup.get_collection(
        str(palace), collection_name="mempalace_drawers", create=True
    )

    ChromaBackend._quarantined_paths.discard(str(palace))

    call_count = {"n": 0}
    real_quarantine = quarantine_stale_hnsw

    def counting_quarantine(*args, **kwargs):
        call_count["n"] += 1
        return real_quarantine(*args, **kwargs)

    import mempalace.backends.chroma as chroma_module

    original = chroma_module.quarantine_stale_hnsw
    chroma_module.quarantine_stale_hnsw = counting_quarantine
    try:
        backend = ChromaBackend()
        # Three sequential opens; quarantine should fire at most once.
        backend.get_collection(str(palace), collection_name="mempalace_drawers", create=False)
        backend.get_collection(str(palace), collection_name="mempalace_drawers", create=False)
        backend.get_collection(str(palace), collection_name="mempalace_drawers", create=False)
    finally:
        chroma_module.quarantine_stale_hnsw = original

    assert call_count["n"] <= 1, (
        f"quarantine_stale_hnsw fired {call_count['n']} times across 3 opens; "
        "expected at most 1 (gated by _quarantined_paths)."
    )
