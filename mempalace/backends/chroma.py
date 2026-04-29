"""ChromaDB-backed MemPalace collection adapter."""

import datetime as _dt
import logging
import os
import sqlite3

import chromadb

from .base import BaseCollection

logger = logging.getLogger(__name__)


def _last_write_per_vector_segment(db_path: str) -> dict[str, float]:
    """Map each VECTOR segment uuid to the epoch-seconds timestamp of the
    most recent embedding insert into its collection's METADATA segment.

    Used by quarantine_stale_hnsw to compare each HNSW dir's mtime against
    its OWN collection's last-write time, not the shared chroma.sqlite3
    mtime. The shared-mtime heuristic produced false positives whenever
    one wing was hot and another was cold — the cold one looked stale on
    every palace open.

    Returns an empty dict on any sqlite error; callers should treat that
    as "don't quarantine anything" rather than "quarantine everything".
    """
    out: dict[str, float] = {}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        logger.exception("Could not open %s for segment metadata read", db_path)
        return out
    try:
        rows = conn.execute(
            """
            SELECT
              vec.id AS vector_id,
              COALESCE(
                (SELECT MAX(strftime('%s', e.created_at))
                 FROM embeddings e
                 WHERE e.segment_id = meta.id),
                0
              ) AS last_write_epoch
            FROM segments vec
            JOIN segments meta
              ON meta.collection = vec.collection
             AND meta.scope = 'METADATA'
            WHERE vec.scope = 'VECTOR'
            """
        ).fetchall()
        for vector_id, last_write_epoch in rows:
            try:
                out[vector_id] = float(last_write_epoch or 0)
            except (TypeError, ValueError):
                out[vector_id] = 0.0
    except sqlite3.Error:
        logger.exception("Could not query segments/embeddings in %s", db_path)
    finally:
        conn.close()
    return out


def quarantine_stale_hnsw(palace_path: str, stale_seconds: float = 86400.0) -> list[str]:
    """Rename HNSW segment dirs that are stale relative to their OWN
    collection's most recent write.

    When a ChromaDB 1.5.x PersistentClient opens a palace whose on-disk
    HNSW segment is significantly older than its collection's metadata,
    the Rust graph-walk can dereference dangling neighbor pointers for
    entries that exist in metadata but not in the HNSW index, and segfault
    in a background thread on the next ``count()`` or ``query(...)`` call.

    First-pass attempt at this fix (commit 4eb6122 on personal-v2, reverted)
    used the SHARED chroma.sqlite3 mtime as the staleness reference. That
    heuristic was unsound for multi-collection palaces: any write to
    collection A bumped chroma.sqlite3.mtime and made every other
    collection's HNSW look stale. With ~50 wings, cold wings got
    quarantined every time hot wings were written. See diary entry
    mempalace-quarantine-regression-2026-04-28 for the incident.

    This pass uses per-collection mtimes by joining segments → embeddings.
    A vector segment is stale only when its OWN collection's metadata has
    been written substantially more recently than the segment's HNSW file.
    Threshold raised to 24h (was 1h) — the real-world failure mode in the
    upstream issue (#823, neo-cortex-mcp#2, chroma-core#2594) was
    multi-day drift from interrupted writes, not minute-scale flush lag.

    Orphan UUIDs (segment dir on disk but no row in chroma's segments
    table) are skipped — they're leftovers from prior segment generations
    that chroma already rotated past, not corruption to recover from.

    Args:
        palace_path: directory containing chroma.sqlite3 + segment dirs
        stale_seconds: minimum gap (collection's last write - HNSW mtime)
                       to treat a segment as stale. Default 24h.

    Returns:
        Quarantine paths created, empty list if nothing was renamed.
    """
    db_path = os.path.join(palace_path, "chroma.sqlite3")
    if not os.path.isfile(db_path):
        return []

    last_write = _last_write_per_vector_segment(db_path)
    if not last_write:
        return []

    moved: list[str] = []
    try:
        entries = os.listdir(palace_path)
    except OSError:
        return []

    for name in entries:
        if "-" not in name or name.startswith(".") or ".drift-" in name or ".empty-stub-" in name:
            continue
        seg_dir = os.path.join(palace_path, name)
        if not os.path.isdir(seg_dir):
            continue
        # Skip orphans: dir present, but no row in segments table for this
        # uuid. Old code quarantined these too, which produced the orphan
        # accumulation we saw on personal-v2.
        coll_last_write = last_write.get(name)
        if coll_last_write is None or coll_last_write <= 0:
            continue
        hnsw_bin = os.path.join(seg_dir, "data_level0.bin")
        if not os.path.isfile(hnsw_bin):
            continue
        try:
            hnsw_mtime = os.path.getmtime(hnsw_bin)
        except OSError:
            continue
        gap = coll_last_write - hnsw_mtime
        if gap < stale_seconds:
            continue
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        target = f"{seg_dir}.drift-{stamp}"
        try:
            os.rename(seg_dir, target)
            moved.append(target)
            logger.warning(
                "Quarantined HNSW segment %s: collection's last write was %.0fs newer "
                "than HNSW data_level0.bin; renamed to %s",
                seg_dir,
                gap,
                target,
            )
        except OSError:
            logger.exception("Failed to quarantine stale HNSW segment %s", seg_dir)
    return moved


def _fix_blob_seq_ids(palace_path: str):
    """Fix ChromaDB 0.6.x -> 1.5.x migration bug: BLOB seq_ids -> INTEGER.

    ChromaDB 0.6.x stored seq_id as big-endian 8-byte BLOBs. ChromaDB 1.5.x
    expects INTEGER. The auto-migration doesn't convert existing rows, causing
    the Rust compactor to crash with "mismatched types; Rust type u64 (as SQL
    type INTEGER) is not compatible with SQL type BLOB".

    Must run BEFORE PersistentClient is created (the compactor fires on init).
    """
    db_path = os.path.join(palace_path, "chroma.sqlite3")
    if not os.path.isfile(db_path):
        return
    try:
        with sqlite3.connect(db_path) as conn:
            for table in ("embeddings", "max_seq_id"):
                try:
                    rows = conn.execute(
                        f"SELECT rowid, seq_id FROM {table} WHERE typeof(seq_id) = 'blob'"
                    ).fetchall()
                except sqlite3.OperationalError:
                    continue
                if not rows:
                    continue
                updates = [(int.from_bytes(blob, byteorder="big"), rowid) for rowid, blob in rows]
                conn.executemany(f"UPDATE {table} SET seq_id = ? WHERE rowid = ?", updates)
                logger.info("Fixed %d BLOB seq_ids in %s", len(updates), table)
            conn.commit()
    except Exception:
        logger.exception("Could not fix BLOB seq_ids in %s", db_path)


class ChromaCollection(BaseCollection):
    """Thin adapter over a ChromaDB collection."""

    def __init__(self, collection):
        self._collection = collection

    def add(self, *, documents, ids, metadatas=None):
        self._collection.add(documents=documents, ids=ids, metadatas=metadatas)

    def upsert(self, *, documents, ids, metadatas=None):
        self._collection.upsert(documents=documents, ids=ids, metadatas=metadatas)

    def update(self, **kwargs):
        self._collection.update(**kwargs)

    def query(self, **kwargs):
        return self._collection.query(**kwargs)

    def get(self, **kwargs):
        return self._collection.get(**kwargs)

    def delete(self, **kwargs):
        self._collection.delete(**kwargs)

    def count(self):
        return self._collection.count()


class ChromaBackend:
    """Factory for MemPalace's default ChromaDB backend."""

    def __init__(self):
        # Per-instance client cache: palace_path -> chromadb.PersistentClient
        self._clients: dict = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _client(self, palace_path: str):
        """Return a cached PersistentClient for *palace_path*, creating one if needed."""
        if palace_path not in self._clients:
            if os.environ.get("MEMPALACE_HNSW_QUARANTINE_ENABLE"):
                quarantine_stale_hnsw(palace_path)
            _fix_blob_seq_ids(palace_path)
            self._clients[palace_path] = chromadb.PersistentClient(path=palace_path)
        return self._clients[palace_path]

    # ------------------------------------------------------------------
    # Public static helpers (for callers that manage their own caching)
    # ------------------------------------------------------------------

    @staticmethod
    def make_client(palace_path: str):
        """Create and return a fresh PersistentClient (fix BLOB seq_ids first).

        Intended for long-lived callers (e.g. mcp_server) that keep their own
        inode/mtime-based client cache.
        """
        if os.environ.get("MEMPALACE_HNSW_QUARANTINE_ENABLE"):
            quarantine_stale_hnsw(palace_path)
        _fix_blob_seq_ids(palace_path)
        return chromadb.PersistentClient(path=palace_path)

    @staticmethod
    def backend_version() -> str:
        """Return the installed chromadb package version string."""
        return chromadb.__version__

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------

    def get_collection(self, palace_path: str, collection_name: str, create: bool = False):
        if not create and not os.path.isdir(palace_path):
            raise FileNotFoundError(palace_path)

        if create:
            os.makedirs(palace_path, exist_ok=True)
            try:
                os.chmod(palace_path, 0o700)
            except (OSError, NotImplementedError):
                pass

        client = self._client(palace_path)
        if create:
            collection = client.get_or_create_collection(
                collection_name, metadata={"hnsw:space": "cosine"}
            )
        else:
            collection = client.get_collection(collection_name)
        return ChromaCollection(collection)

    def get_or_create_collection(
        self, palace_path: str, collection_name: str
    ) -> "ChromaCollection":
        """Shorthand for get_collection(..., create=True)."""
        return self.get_collection(palace_path, collection_name, create=True)

    def delete_collection(self, palace_path: str, collection_name: str) -> None:
        """Delete *collection_name* from the palace at *palace_path*."""
        self._client(palace_path).delete_collection(collection_name)

    def create_collection(
        self, palace_path: str, collection_name: str, hnsw_space: str = "cosine"
    ) -> "ChromaCollection":
        """Create (not get-or-create) *collection_name* with cosine HNSW space."""
        collection = self._client(palace_path).create_collection(
            collection_name, metadata={"hnsw:space": hnsw_space}
        )
        return ChromaCollection(collection)
