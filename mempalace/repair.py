"""
repair.py — Scan, prune corrupt entries, and rebuild HNSW index
================================================================

When ChromaDB's HNSW index accumulates duplicate entries (from repeated
add() calls with the same ID), link_lists.bin can grow unbounded —
terabytes on large palaces — eventually causing segfaults.

This module provides four operations:

  status  — compare sqlite vs HNSW element counts (read-only health check)
  scan    — find every corrupt/unfetchable ID in the palace
  prune   — delete only the corrupt IDs (surgical)
  rebuild — extract all drawers, delete the collection, recreate with
            correct HNSW settings, and upsert everything back

The rebuild backs up ONLY chroma.sqlite3 (the source of truth), not the
full palace directory — so it works even when link_lists.bin is bloated.

Usage (standalone):
    python -m mempalace.repair status
    python -m mempalace.repair scan [--wing X]
    python -m mempalace.repair prune --confirm
    python -m mempalace.repair rebuild

Usage (from CLI):
    mempalace repair
    mempalace repair-scan [--wing X]
    mempalace repair-prune --confirm
"""

import argparse
import os
import shutil
import sqlite3
import time
from datetime import datetime
from typing import Optional

from .backends.chroma import ChromaBackend, hnsw_capacity_status


COLLECTION_NAME = "mempalace_drawers"


def _get_palace_path():
    """Resolve palace path from config."""
    try:
        from .config import MempalaceConfig

        return MempalaceConfig().palace_path
    except Exception:
        default = os.path.join(os.path.expanduser("~"), ".mempalace", "palace")
        return default


def _paginate_ids(col, where=None):
    """Pull all IDs in a collection using pagination."""
    ids = []
    page = 1000
    offset = 0
    while True:
        try:
            r = col.get(where=where, include=[], limit=page, offset=offset)
        except Exception:
            try:
                r = col.get(where=where, include=[], limit=page)
                new_ids = [i for i in r["ids"] if i not in set(ids)]
                if not new_ids:
                    break
                ids.extend(new_ids)
                offset += len(new_ids)
                continue
            except Exception:
                break
        n = len(r["ids"]) if r["ids"] else 0
        if n == 0:
            break
        ids.extend(r["ids"])
        offset += n
        if n < page:
            break
    return ids


def scan_palace(palace_path=None, only_wing=None):
    """Scan the palace for corrupt/unfetchable IDs.

    Probes in batches of 100, falls back to per-ID on failure.
    Writes corrupt_ids.txt to the palace directory for the prune step.

    Returns (good_set, bad_set).
    """
    palace_path = palace_path or _get_palace_path()
    print(f"\n  Palace: {palace_path}")
    print("  Loading...")

    col = ChromaBackend().get_collection(palace_path, COLLECTION_NAME)

    where = {"wing": only_wing} if only_wing else None
    total = col.count()
    print(f"  Collection: {COLLECTION_NAME}, total: {total:,}")
    if only_wing:
        print(f"  Scanning wing: {only_wing}")

    print("\n  Step 1: listing all IDs...")
    t0 = time.time()
    all_ids = _paginate_ids(col, where=where)
    print(f"  Found {len(all_ids):,} IDs in {time.time() - t0:.1f}s\n")

    if not all_ids:
        print("  Nothing to scan.")
        return set(), set()

    print("  Step 2: probing each ID (batches of 100)...")
    t0 = time.time()
    good_set = set()
    bad_set = set()
    batch = 100

    for i in range(0, len(all_ids), batch):
        chunk = all_ids[i : i + batch]
        try:
            r = col.get(ids=chunk, include=["documents"])
            for got in r["ids"]:
                good_set.add(got)
            for mid in chunk:
                if mid not in good_set:
                    bad_set.add(mid)
        except Exception:
            for sid in chunk:
                try:
                    r = col.get(ids=[sid], include=["documents"])
                    if r["ids"]:
                        good_set.add(sid)
                    else:
                        bad_set.add(sid)
                except Exception:
                    bad_set.add(sid)

        if (i // batch) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + batch) / max(elapsed, 0.01)
            eta = (len(all_ids) - i - batch) / max(rate, 0.01)
            print(
                f"    {i + batch:>6}/{len(all_ids):>6}  "
                f"good={len(good_set):>6}  bad={len(bad_set):>6}  "
                f"eta={eta:.0f}s"
            )

    print(f"\n  Scan complete in {time.time() - t0:.1f}s")
    print(f"  GOOD: {len(good_set):,}")
    print(f"  BAD:  {len(bad_set):,}  ({len(bad_set) / max(len(all_ids), 1) * 100:.1f}%)")

    bad_file = os.path.join(palace_path, "corrupt_ids.txt")
    with open(bad_file, "w") as f:
        for bid in sorted(bad_set):
            f.write(bid + "\n")
    print(f"\n  Bad IDs written to: {bad_file}")
    return good_set, bad_set


def prune_corrupt(palace_path=None, confirm=False):
    """Delete corrupt IDs listed in corrupt_ids.txt."""
    palace_path = palace_path or _get_palace_path()
    bad_file = os.path.join(palace_path, "corrupt_ids.txt")

    if not os.path.exists(bad_file):
        print("  No corrupt_ids.txt found — run scan first.")
        return

    with open(bad_file) as f:
        bad_ids = [line.strip() for line in f if line.strip()]
    print(f"  {len(bad_ids):,} corrupt IDs queued for deletion")

    if not confirm:
        print("\n  DRY RUN — no deletions performed.")
        print("  Re-run with --confirm to actually delete.")
        return

    col = ChromaBackend().get_collection(palace_path, COLLECTION_NAME)
    before = col.count()
    print(f"  Collection size before: {before:,}")

    batch = 100
    deleted = 0
    failed = 0
    for i in range(0, len(bad_ids), batch):
        chunk = bad_ids[i : i + batch]
        try:
            col.delete(ids=chunk)
            deleted += len(chunk)
        except Exception:
            for sid in chunk:
                try:
                    col.delete(ids=[sid])
                    deleted += 1
                except Exception:
                    failed += 1
        if (i // batch) % 20 == 0:
            print(f"    deleted {deleted}/{len(bad_ids)}  (failed: {failed})")

    after = col.count()
    print(f"\n  Deleted: {deleted:,}")
    print(f"  Failed:  {failed:,}")
    print(f"  Collection size: {before:,} → {after:,}")


# ChromaDB's ``collection.get()`` enforces an internal default ``limit``
# of 10 000 rows when the caller does not pass one. We pass an explicit
# ``limit=batch_size`` below, but the underlying segment also caps reads
# during stale/quarantined-HNSW recovery flows: extraction silently stops
# at exactly 10 000 even on palaces with many more rows. Refusing to
# overwrite when this exact value comes back is the simplest signal we
# can detect without depending on chromadb internals.
CHROMADB_DEFAULT_GET_LIMIT = 10_000


class TruncationDetected(Exception):
    """Raised by :func:`check_extraction_safety` when extraction looks short.

    Carries the human-readable abort message so callers (CLI ``cmd_repair``,
    ``rebuild_index``) can print and exit consistently without re-deriving
    the wording.
    """

    def __init__(self, message: str, sqlite_count: "int | None", extracted: int):
        super().__init__(message)
        self.message = message
        self.sqlite_count = sqlite_count
        self.extracted = extracted


def check_extraction_safety(
    palace_path: str, extracted: int, confirm_truncation_ok: bool = False
) -> None:
    """Cross-check that ``extracted`` matches the SQLite ground truth.

    Two signals trip the guard:

    1. **Strong** — ``chroma.sqlite3`` reports more drawers than were
       extracted. This is the user-reported #1208 case: 67 580 on disk,
       10 000 came back through the chromadb collection layer, repair
       would have destroyed the difference.
    2. **Weak** — extracted count equals exactly ``CHROMADB_DEFAULT_GET_LIMIT``
       AND the SQLite check couldn't run (schema drift, locked file).
       Hitting the chromadb default ``get()`` cap exactly is suspicious
       enough to refuse without explicit acknowledgement.

    Raises :class:`TruncationDetected` with a printable message when the
    guard fires. Does nothing on safe extractions or when
    ``confirm_truncation_ok`` is set.
    """
    if confirm_truncation_ok:
        return

    sqlite_count = sqlite_drawer_count(palace_path)
    cap_signal = extracted == CHROMADB_DEFAULT_GET_LIMIT

    if sqlite_count is not None and sqlite_count > extracted:
        loss = sqlite_count - extracted
        pct = 100 * loss / sqlite_count
        message = (
            f"\n  ABORT: chroma.sqlite3 reports {sqlite_count:,} drawers but only {extracted:,}\n"
            "  came back through the chromadb collection layer. The segment metadata is\n"
            "  stale (often after manual HNSW quarantine) — proceeding would silently\n"
            f"  destroy {loss:,} drawers (~{pct:.0f}%).\n"
            "\n"
            "  Recovery options:\n"
            "    1. Restore from your most recent palace backup, then re-mine.\n"
            "    2. Direct-extract from chroma.sqlite3 (rows are still on disk) and\n"
            "       rebuild the palace from source files.\n"
            "    3. If you have independently confirmed the palace really contains only\n"
            f"       {extracted:,} drawers, re-run with --confirm-truncation-ok.\n"
        )
        raise TruncationDetected(message, sqlite_count, extracted)

    if cap_signal and sqlite_count is None:
        message = (
            f"\n  ABORT: extracted exactly {CHROMADB_DEFAULT_GET_LIMIT:,} drawers, which matches\n"
            "  ChromaDB's internal default get() limit. The on-disk SQLite count couldn't\n"
            "  be cross-checked from this Python context, so we can't tell whether the\n"
            f"  palace genuinely holds {CHROMADB_DEFAULT_GET_LIMIT:,} rows or whether extraction was\n"
            "  silently capped. Refusing to overwrite the palace.\n"
            "\n"
            "  If you have independently confirmed (e.g. via direct sqlite3 query) that\n"
            f"  the palace really contains exactly {CHROMADB_DEFAULT_GET_LIMIT:,} drawers, re-run with\n"
            "  --confirm-truncation-ok.\n"
        )
        raise TruncationDetected(message, sqlite_count, extracted)


def sqlite_drawer_count(palace_path: str) -> "int | None":
    """Count rows in ``chroma.sqlite3.embeddings`` for the drawers collection.

    Used as an independent ground-truth check against the chromadb
    collection-layer ``count()`` / ``get()``: when the on-disk SQLite
    row count exceeds the extraction count, the segment metadata is
    stale and repair would destroy the difference.

    Returns ``None`` when the schema isn't readable (chromadb version
    drift, missing tables, locked file). Callers treat ``None`` as
    "unknown" and fall back to the cap-detection check.
    """
    sqlite_path = os.path.join(palace_path, "chroma.sqlite3")
    if not os.path.exists(sqlite_path):
        return None
    try:
        import sqlite3

        conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
        try:
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM embeddings e
                JOIN segments s ON e.segment_id = s.id
                JOIN collections c ON s.collection = c.id
                WHERE c.name = ?
                """,
                (COLLECTION_NAME,),
            ).fetchone()
            return int(row[0]) if row and row[0] is not None else None
        finally:
            conn.close()
    except Exception:
        # chromadb schema differs by version (segments / collections column
        # names occasionally rename). Silent fallback is correct here —
        # the cap-detection check still catches the user-reported case.
        return None


def rebuild_index(palace_path=None, confirm_truncation_ok: bool = False):
    """Rebuild the HNSW index from scratch.

    1. Extract all drawers via ChromaDB get()
    2. Cross-check against the SQLite ground truth (#1208 guard)
    3. Back up ONLY chroma.sqlite3 (not the bloated HNSW files)
    4. Delete and recreate the collection with hnsw:space=cosine
    5. Upsert all drawers back

    ``confirm_truncation_ok`` overrides the safety guard from step 2.
    Set to ``True`` only when you have independently verified that the
    palace genuinely contains exactly the extracted number of drawers
    (typically only a concern for palaces sized at exactly 10 000 rows).
    """
    palace_path = palace_path or _get_palace_path()

    if not os.path.isdir(palace_path):
        print(f"\n  No palace found at {palace_path}")
        return

    print(f"\n{'=' * 55}")
    print("  MemPalace Repair — Index Rebuild")
    print(f"{'=' * 55}\n")
    print(f"  Palace: {palace_path}")

    backend = ChromaBackend()
    try:
        col = backend.get_collection(palace_path, COLLECTION_NAME)
        total = col.count()
    except Exception as e:
        print(f"  Error reading palace: {e}")
        print("  Palace may need to be re-mined from source files.")
        return

    print(f"  Drawers found: {total}")

    if total == 0:
        print("  Nothing to repair.")
        return

    # Extract all drawers in batches
    print("\n  Extracting drawers...")
    batch_size = 5000
    all_ids = []
    all_docs = []
    all_metas = []
    offset = 0
    while offset < total:
        batch = col.get(limit=batch_size, offset=offset, include=["documents", "metadatas"])
        if not batch["ids"]:
            break
        all_ids.extend(batch["ids"])
        all_docs.extend(batch["documents"])
        all_metas.extend(batch["metadatas"])
        offset += len(batch["ids"])
    print(f"  Extracted {len(all_ids)} drawers")

    # ── #1208 guard ──────────────────────────────────────────────────
    # Refuse to ``delete_collection`` + rebuild when extraction looks
    # short of the SQLite ground truth (or when extraction == chromadb
    # default get() cap and the SQLite check couldn't run).
    try:
        check_extraction_safety(palace_path, len(all_ids), confirm_truncation_ok)
    except TruncationDetected as e:
        print(e.message)
        return

    # Back up ONLY the SQLite database, not the bloated HNSW files
    sqlite_path = os.path.join(palace_path, "chroma.sqlite3")
    backup_path = sqlite_path + ".backup"
    if os.path.exists(sqlite_path):
        print(f"  Backing up chroma.sqlite3 ({os.path.getsize(sqlite_path) / 1e6:.0f} MB)...")
        shutil.copy2(sqlite_path, backup_path)
        print(f"  Backup: {backup_path}")

    # Rebuild with correct HNSW settings.
    #
    # ``metadata_overrides`` overrides the ``_HNSW_BLOAT_GUARD`` defaults
    # (sync_threshold=50_000, batch_size=50_000). The bloat guard exists
    # to prevent resize-cycle bloat under incremental writes; on a
    # one-shot bulk rebuild it instead leaves the trailing buffer
    # between the last 50k flush boundary and ``len(all_ids)`` unflushed,
    # silently stranding up to ~50k records in ``embeddings_queue``.
    # 1_000 keeps flush frequency moderate without re-introducing bloat.
    print("  Rebuilding collection with hnsw:space=cosine...")
    backend.delete_collection(palace_path, COLLECTION_NAME)
    new_col = backend.create_collection(
        palace_path,
        COLLECTION_NAME,
        metadata_overrides={
            "hnsw:batch_size": 1_000,
            "hnsw:sync_threshold": 1_000,
        },
    )

    filed = 0
    try:
        for i in range(0, len(all_ids), batch_size):
            batch_ids = all_ids[i : i + batch_size]
            batch_docs = all_docs[i : i + batch_size]
            batch_metas = all_metas[i : i + batch_size]
            new_col.upsert(documents=batch_docs, ids=batch_ids, metadatas=batch_metas)
            filed += len(batch_ids)
            print(f"  Re-filed {filed}/{len(all_ids)} drawers...")
    except Exception as e:
        print(f"\n  ERROR during rebuild: {e}")
        print(f"  Only {filed}/{len(all_ids)} drawers were re-filed.")
        if os.path.exists(backup_path):
            print(f"  Restoring from backup: {backup_path}")
            backend.delete_collection(palace_path, COLLECTION_NAME)
            shutil.copy2(backup_path, sqlite_path)
            print("  Backup restored. Palace is back to pre-repair state.")
        else:
            print("  No backup available. Re-mine from source files to recover.")
        raise

    # ── Post-rebuild verification ─────────────────────────────────────
    # ChromaDB's upsert returns success even when the persistent HNSW
    # silently fails to flush (the original bug class this fix targets).
    # Cross-check the new segment's ``hnsw_capacity_status`` against
    # ``len(all_ids)`` and refuse to declare success if diverged. The
    # backup copy of the OLD chroma.sqlite3 is preserved at
    # ``backup_path`` so the operator can restore manually.
    cap = hnsw_capacity_status(palace_path, COLLECTION_NAME)
    hnsw_count = cap.get("hnsw_count")
    if cap.get("diverged") or (hnsw_count is not None and abs(hnsw_count - filed) > 1_000):
        print(
            f"\n  ERROR: post-rebuild HNSW divergence detected — "
            f"expected {filed:,}, HNSW reports {hnsw_count}. "
            f"sqlite={cap.get('sqlite_count')}."
        )
        print(f"  Backup of pre-repair chroma.sqlite3 preserved at: {backup_path}")
        print("  To restore: stop any running mempalace process, then")
        print(f"    cp {backup_path} {sqlite_path}")
        print("  Investigate chromadb's HNSW flush behavior before retrying.")
        return

    print(f"\n  Repair complete. {filed} drawers rebuilt.")
    print(f"  HNSW verification: hnsw={hnsw_count:,}, sqlite={cap.get('sqlite_count'):,}.")
    print("  HNSW index is now clean with cosine distance metric.")
    print(f"\n{'=' * 55}\n")


def status(palace_path=None) -> dict:
    """Read-only health check: compare sqlite vs HNSW element counts.

    Catches the #1222 failure mode where chromadb's HNSW segment freezes
    at a stale ``max_elements`` while sqlite keeps accumulating rows.
    Once the divergence is large enough, every tool call segfaults when
    chromadb tries to load the undersized HNSW. Running ``mempalace
    repair-status`` *before* opening the segment lets the operator
    discover the problem without crashing the MCP server.

    The check itself never opens a chromadb client and never imports
    hnswlib — it reads ``chroma.sqlite3`` and ``index_metadata.pickle``
    directly via :func:`mempalace.backends.chroma.hnsw_capacity_status`.

    Returns the capacity-status dict (also printed). Returns a dict with
    ``status="unknown"`` when no palace exists at the given path.
    """
    palace_path = palace_path or _get_palace_path()
    print(f"\n{'=' * 55}")
    print("  MemPalace Repair — Status")
    print(f"{'=' * 55}\n")
    print(f"  Palace: {palace_path}")

    if not os.path.isdir(palace_path):
        print("  No palace found.\n")
        return {"status": "unknown", "message": "no palace at path"}

    drawers = hnsw_capacity_status(palace_path, "mempalace_drawers")
    closets = hnsw_capacity_status(palace_path, "mempalace_closets")

    for label, info in (("drawers", drawers), ("closets", closets)):
        print(f"\n  [{label}]")
        if info["sqlite_count"] is None:
            print("    sqlite count:   (unreadable)")
        else:
            print(f"    sqlite count:   {info['sqlite_count']:,}")
        if info["hnsw_count"] is None:
            print("    hnsw count:     (no flushed metadata yet)")
        else:
            print(f"    hnsw count:     {info['hnsw_count']:,}")
        if info["divergence"] is not None:
            print(f"    divergence:     {info['divergence']:,}")
        marker = "DIVERGED" if info["diverged"] else info["status"].upper()
        print(f"    status:         {marker}")
        if info["message"]:
            print(f"    note:           {info['message']}")

    if drawers["diverged"] or closets["diverged"]:
        print("\n  Recommended: run `mempalace repair` to rebuild the index.")
    print()
    return {"drawers": drawers, "closets": closets}


# ---------------------------------------------------------------------------
# max-seq-id mode: un-poison max_seq_id rows corrupted by the old shim
# ---------------------------------------------------------------------------


def _close_chroma_handles(palace_path: str) -> None:
    """Drop ChromaBackend + chromadb singleton caches so OS mmap handles release."""
    import gc

    try:
        ChromaBackend().close_palace(palace_path)
    except Exception:
        pass
    try:
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception:
        pass
    gc.collect()


class MaxSeqIdVerificationError(RuntimeError):
    """Raised when post-repair detection still sees poisoned rows."""


#: Any ``max_seq_id.seq_id`` above this is unreachable by a real palace.
#: Clean values are bounded by the embeddings_queue's monotonic counter (<1e10
#: in practice), and 2**53 is the float64 exact-integer ceiling. Poisoned
#: values from the 0.6.x shim misinterpreting chromadb 1.5.x's
#: ``b'\x11\x11' + 6 ASCII digits`` format start at ~1.23e18, so anything
#: above the threshold is confidently a shim-poisoning artefact.
MAX_SEQ_ID_SANITY_THRESHOLD = 1 << 53


def _detect_poisoned_max_seq_ids(
    db_path: str,
    *,
    segment: Optional[str] = None,
    threshold: int = MAX_SEQ_ID_SANITY_THRESHOLD,
) -> list[tuple[str, int]]:
    """Return ``[(segment_id, poisoned_seq_id), ...]`` for rows above threshold.

    If ``segment`` is given, the detection is restricted to that segment id
    (still only returning it if it actually exceeds the threshold).
    """
    with sqlite3.connect(db_path) as conn:
        if segment is not None:
            rows = conn.execute(
                "SELECT segment_id, seq_id FROM max_seq_id WHERE segment_id = ? AND seq_id > ?",
                (segment, threshold),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT segment_id, seq_id FROM max_seq_id WHERE seq_id > ?",
                (threshold,),
            ).fetchall()
    return [(str(sid), int(val)) for sid, val in rows]


def _compute_heuristic_seq_id(cur: sqlite3.Cursor, segment_id: str) -> int:
    """Return ``MAX(embeddings.seq_id)`` over the collection owning ``segment_id``.

    Matches the METADATA segment's pre-poison value exactly (its max equals
    the collection-wide embeddings max). For the sibling VECTOR segment the
    value is a few seq_ids ahead of its own pre-poison max; the queue
    treats that as "already consumed", skipping a small window of
    already-indexed embeddings on next subscribe. That is an acceptable
    loss vs. resetting to 0 (which would re-process the entire queue and
    risk HNSW bloat from issue #1046).
    """
    row = cur.execute(
        """
        SELECT MAX(e.seq_id)
        FROM embeddings e
        JOIN segments s ON e.segment_id = s.id
        WHERE s.collection = (
            SELECT collection FROM segments WHERE id = ?
        )
        """,
        (segment_id,),
    ).fetchone()
    if row is None or row[0] is None:
        return 0
    return int(row[0])


def _read_sidecar_seq_ids(sidecar_path: str) -> dict[str, int]:
    """Load ``{segment_id: seq_id}`` from a sidecar DB's ``max_seq_id`` table.

    Rejects sidecar files whose ``max_seq_id.seq_id`` is itself BLOB-typed
    — a sidecar that old predates chromadb's type normalisation and is not
    a trustworthy restoration source.
    """
    if not os.path.isfile(sidecar_path):
        raise FileNotFoundError(f"Sidecar database not found: {sidecar_path}")
    out: dict[str, int] = {}
    with sqlite3.connect(sidecar_path) as conn:
        rows = conn.execute("SELECT segment_id, seq_id, typeof(seq_id) FROM max_seq_id").fetchall()
    for segment_id, seq_id, kind in rows:
        if kind == "blob":
            raise ValueError(
                f"Sidecar has BLOB-typed seq_id for {segment_id}; refusing to use it. "
                "Pass a sidecar that was already migrated to INTEGER rows."
            )
        out[str(segment_id)] = int(seq_id)
    return out


def repair_max_seq_id(
    palace_path: str,
    *,
    segment: Optional[str] = None,
    from_sidecar: Optional[str] = None,
    threshold: int = MAX_SEQ_ID_SANITY_THRESHOLD,
    backup: bool = True,
    dry_run: bool = False,
    assume_yes: bool = False,
) -> dict:
    """Un-poison ``max_seq_id`` rows corrupted by ``_fix_blob_seq_ids`` misfire.

    The old shim ran ``int.from_bytes(blob, 'big')`` across every BLOB
    ``max_seq_id.seq_id`` row, including chromadb 1.5.x's native
    ``b'\\x11\\x11' + ASCII digits`` format. That conversion yields a
    ~1.23e18 integer that silently suppresses every subsequent
    ``embeddings_queue`` write for the affected segment. This command
    restores clean values either from a pre-corruption sidecar DB
    (exact) or heuristically (``MAX(embeddings.seq_id)`` over the owning
    collection).
    """
    from .migrate import confirm_destructive_action, contains_palace_database

    palace_path = os.path.abspath(os.path.expanduser(palace_path))
    db_path = os.path.join(palace_path, "chroma.sqlite3")

    result: dict = {
        "palace_path": palace_path,
        "dry_run": dry_run,
        "aborted": False,
        "segment_repaired": [],
        "before": {},
        "after": {},
        "backup": None,
    }

    print(f"\n{'=' * 55}")
    print("  MemPalace Repair — max_seq_id Un-poison")
    print(f"{'=' * 55}\n")
    print(f"  Palace:  {palace_path}")
    if segment:
        print(f"  Segment: {segment}")
    if from_sidecar:
        print(f"  Sidecar: {from_sidecar}")

    if not os.path.isdir(palace_path):
        print(f"  No palace found at {palace_path}")
        result["aborted"] = True
        result["reason"] = "palace-missing"
        return result
    if not contains_palace_database(palace_path):
        print(f"  No palace database at {palace_path}")
        result["aborted"] = True
        result["reason"] = "db-missing"
        return result

    poisoned = _detect_poisoned_max_seq_ids(db_path, segment=segment, threshold=threshold)
    if not poisoned:
        print("  No poisoned max_seq_id rows detected. Nothing to do.")
        print(f"\n{'=' * 55}\n")
        return result

    sidecar_map: dict[str, int] = {}
    if from_sidecar:
        sidecar_map = _read_sidecar_seq_ids(from_sidecar)

    plan: list[tuple[str, int, int]] = []
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        for seg_id, old_val in poisoned:
            if from_sidecar:
                if seg_id not in sidecar_map:
                    print(f"  Skipped segment {seg_id}: no sidecar entry")
                    continue
                new_val = sidecar_map[seg_id]
            else:
                new_val = _compute_heuristic_seq_id(cur, seg_id)
            plan.append((seg_id, old_val, new_val))
            result["before"][seg_id] = old_val
            result["after"][seg_id] = new_val

    print()
    print("  Report")
    print(f"    poisoned rows        {len(poisoned):>6}")
    print(f"    planned repairs      {len(plan):>6}")
    source = "sidecar" if from_sidecar else "heuristic (collection MAX)"
    print(f"    clean-value source   {source}")
    for seg_id, old_val, new_val in plan:
        print(f"    {seg_id}  {old_val}  →  {new_val}")

    if dry_run:
        print("\n  DRY RUN — no rows modified.\n" + "=" * 55 + "\n")
        return result

    if not plan:
        print("  No actionable repairs.")
        print(f"\n{'=' * 55}\n")
        return result

    if not confirm_destructive_action("Repair max_seq_id", palace_path, assume_yes=assume_yes):
        result["aborted"] = True
        result["reason"] = "user-aborted"
        return result

    if backup:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = os.path.join(palace_path, f"chroma.sqlite3.max-seq-id-backup-{timestamp}")
        shutil.copy2(db_path, backup_path)
        result["backup"] = backup_path
        print(f"  Backup:  {backup_path}")

    _close_chroma_handles(palace_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute("BEGIN")
        try:
            conn.executemany(
                "UPDATE max_seq_id SET seq_id = ? WHERE segment_id = ?",
                [(new_val, seg_id) for seg_id, _old, new_val in plan],
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    remaining = _detect_poisoned_max_seq_ids(db_path, segment=segment, threshold=threshold)
    if remaining:
        raise MaxSeqIdVerificationError(
            f"Post-repair detection still found {len(remaining)} poisoned row(s): "
            f"{[sid for sid, _ in remaining]}. Backup at {result['backup']}."
        )

    result["segment_repaired"] = [seg_id for seg_id, _old, _new in plan]
    print(f"\n  Repair complete. {len(plan)} row(s) restored.")
    print(f"  Backup:  {result['backup'] or '(skipped)'}")
    print(f"\n{'=' * 55}\n")
    return result


# ---------------------------------------------------------------------------
# flush-trailing-buffer mode: drain unflushed records into HNSW in place
# ---------------------------------------------------------------------------


def recover_unflushed_buffer(
    palace_path: Optional[str] = None,
    *,
    flush_threshold: int = 100,
    restore_threshold: bool = True,
    quiet: bool = False,
) -> dict:
    """Drain a queue-stranded HNSW segment without a full rebuild.

    Symptom: ``repair-status`` reports DIVERGED with a fixed gap that
    doesn't shrink across MCP server restarts. The drawers segment's
    ``max_seq_id`` row trails ``embeddings_queue.max(seq_id)`` by the
    exact gap size, and ``index_metadata.pickle.total_elements_added``
    is stuck at the last 50_000-multiple record count (e.g. 100_000
    out of 142_172).

    Cause: ``_HNSW_BLOAT_GUARD`` sets ``hnsw:sync_threshold=50_000`` on
    the drawers collection. When a bulk write (e.g. ``mempalace repair``
    pre-fix) ends at a count that is not a multiple of 50_000, the
    trailing buffer between the last flush and end-of-write sits in
    ``embeddings_queue`` indefinitely. The chromadb consumer pulls from
    the queue only when its in-memory ``_curr_batch`` reaches
    ``batch_size``; with batch_size=50_000, queues holding <50_000
    pending records never drain on reopen.

    Recovery is two-session:
      1. Open the collection, modify ``hnsw:sync_threshold`` and
         ``hnsw:batch_size`` to ``flush_threshold`` (default 100).
         ``modify()`` persists the new config to chroma.sqlite3 but the
         in-memory segment instance keeps its old config.
      2. Close. Reopen the palace. The segment instance reloads with
         the new low thresholds. Its ``_backfill`` consumes the queue
         and flushes within ``flush_threshold`` records per batch,
         draining the trailing buffer into HNSW.
      3. Verify by re-reading the segment pickle.
      4. Optionally (default on) restore the original sync_threshold so
         the bloat guard remains in place for steady-state operation.

    No data is destroyed by this routine. The only mutation is to the
    drawers collection's HNSW config (which we restore on exit) and
    HNSW persisting records that ``submit_embeddings`` already accepted.
    """
    palace_path = palace_path or _get_palace_path()
    palace_path = os.path.abspath(os.path.expanduser(palace_path))

    if quiet:
        def print(*_a, **_k):  # noqa: A001 — deliberate local shadow for quiet mode
            return None

    print(f"\n{'=' * 55}")
    print("  MemPalace Repair — Flush Trailing Buffer")
    print(f"{'=' * 55}\n")
    print(f"  Palace:           {palace_path}")
    print(f"  Flush threshold:  {flush_threshold}")
    print(f"  Restore on exit:  {restore_threshold}")

    if not os.path.isdir(palace_path):
        print(f"  No palace at {palace_path}")
        return {"aborted": True, "reason": "palace-missing"}

    before = hnsw_capacity_status(palace_path, COLLECTION_NAME)
    print(
        f"\n  Before:  sqlite={before.get('sqlite_count')}  "
        f"hnsw={before.get('hnsw_count')}  "
        f"divergence={before.get('divergence')}  "
        f"status={before.get('status')}"
    )
    if not before.get("diverged"):
        print("  Already converged — nothing to flush.")
        return {"aborted": False, "before": before, "after": before, "noop": True}

    # ── Session A: lower thresholds via modify() ──────────────────────
    backend = ChromaBackend()
    print("\n  Session A: lowering hnsw:sync_threshold & hnsw:batch_size...")
    try:
        col = backend.get_collection(palace_path, COLLECTION_NAME)
    except Exception as e:
        print(f"  Could not open collection: {e}")
        return {"aborted": True, "reason": "open-failed"}

    try:
        col._collection.modify(
            configuration={
                "hnsw": {
                    "sync_threshold": flush_threshold,
                    "batch_size": flush_threshold,
                }
            }
        )
    except Exception as e:
        print(f"  modify() failed: {e}")
        return {"aborted": True, "reason": "modify-failed"}

    # Drop client + chromadb singleton caches so the next get_collection
    # reads the freshly written config from sqlite instead of reusing
    # the in-memory segment with stale thresholds.
    _close_chroma_handles(palace_path)
    print("  Session A: complete. Closed handles.")

    # ── Session B: reopen so the segment reloads with new thresholds ──
    # Touch the collection once via a count() call. This forces
    # segment lazy-init, which subscribes to embeddings_queue and
    # backfills the pending records. With batch_size=flush_threshold,
    # _apply_batch fires every flush_threshold records and persists.
    print("\n  Session B: reopening to drain queue...")
    backend = ChromaBackend()
    try:
        col = backend.get_collection(palace_path, COLLECTION_NAME)
        live_count = col.count()
        print(f"  col.count()={live_count}")
    except Exception as e:
        print(f"  reopen failed: {e}")
        return {"aborted": True, "reason": "reopen-failed"}

    # Force a final flush by adding+deleting a single nudge record. The
    # nudge runs through submit_embeddings → _notify_one →
    # _write_records → _apply_batch, which crosses the (low) batch_size
    # threshold and persists. We pass a synthetic embedding so we don't
    # need to invoke the embedding function.
    import numpy as np

    nudge_id = "_recover_unflushed_buffer_nudge"
    # Pull dimensionality from an existing record's metadata if we can;
    # else fall back to 384 (sentence-transformers default), since the
    # nudge is deleted immediately and dim only needs to match for the
    # add-then-delete cycle to succeed.
    dim = 384
    try:
        sample = col._collection.get(limit=1, include=["embeddings"])
        if sample and sample.get("embeddings") and len(sample["embeddings"]) > 0:
            emb = sample["embeddings"][0]
            if emb is not None:
                dim = len(emb)
    except Exception:
        pass
    rng = np.random.default_rng(0)
    nudge_emb = rng.standard_normal(dim).astype("float32").tolist()

    try:
        col._collection.add(ids=[nudge_id], embeddings=[nudge_emb])
        col._collection.delete(ids=[nudge_id])
        print(f"  Nudge cycle complete (dim={dim}).")
    except Exception as e:
        print(f"  Nudge failed (continuing): {e}")

    _close_chroma_handles(palace_path)

    after = hnsw_capacity_status(palace_path, COLLECTION_NAME)
    print(
        f"\n  After:   sqlite={after.get('sqlite_count')}  "
        f"hnsw={after.get('hnsw_count')}  "
        f"divergence={after.get('divergence')}  "
        f"status={after.get('status')}"
    )

    # ── Optional Session C: restore the original sync_threshold ───────
    if restore_threshold:
        print("\n  Session C: restoring sync_threshold to bloat-guard default...")
        try:
            from .backends.chroma import _hnsw_metadata_for

            guard = _hnsw_metadata_for(COLLECTION_NAME)
            backend2 = ChromaBackend()
            col2 = backend2.get_collection(palace_path, COLLECTION_NAME)
            col2._collection.modify(
                configuration={
                    "hnsw": {
                        "sync_threshold": guard["hnsw:sync_threshold"],
                        "batch_size": guard["hnsw:batch_size"],
                    }
                }
            )
            print(f"  Restored: {guard}")
            _close_chroma_handles(palace_path)
        except Exception as e:
            print(f"  Restore failed (recovery succeeded; rerun with --no-restore-threshold to skip): {e}")

    print(f"\n{'=' * 55}\n")
    return {"aborted": False, "before": before, "after": after, "noop": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MemPalace repair tools")
    p.add_argument("command", choices=["status", "scan", "prune", "rebuild"])
    p.add_argument("--palace", default=None, help="Palace directory path")
    p.add_argument("--wing", default=None, help="Scan only this wing")
    p.add_argument("--confirm", action="store_true", help="Actually delete corrupt IDs")
    args = p.parse_args()

    path = os.path.expanduser(args.palace) if args.palace else None

    if args.command == "status":
        status(palace_path=path)
    elif args.command == "scan":
        scan_palace(palace_path=path, only_wing=args.wing)
    elif args.command == "prune":
        prune_corrupt(palace_path=path, confirm=args.confirm)
    elif args.command == "rebuild":
        rebuild_index(palace_path=path)
