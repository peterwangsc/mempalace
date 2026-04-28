import os
import sqlite3

import chromadb
import pytest

from mempalace.backends.chroma import (
    ChromaBackend,
    ChromaCollection,
    _fix_blob_seq_ids,
    quarantine_stale_hnsw,
)


class _FakeCollection:
    def __init__(self):
        self.calls = []

    def add(self, **kwargs):
        self.calls.append(("add", kwargs))

    def upsert(self, **kwargs):
        self.calls.append(("upsert", kwargs))

    def query(self, **kwargs):
        self.calls.append(("query", kwargs))
        return {"kind": "query"}

    def get(self, **kwargs):
        self.calls.append(("get", kwargs))
        return {"kind": "get"}

    def delete(self, **kwargs):
        self.calls.append(("delete", kwargs))

    def count(self):
        self.calls.append(("count", {}))
        return 7


def test_chroma_collection_delegates_methods():
    fake = _FakeCollection()
    collection = ChromaCollection(fake)

    collection.add(documents=["d"], ids=["1"], metadatas=[{"wing": "w"}])
    collection.upsert(documents=["u"], ids=["2"], metadatas=[{"room": "r"}])
    assert collection.query(query_texts=["q"]) == {"kind": "query"}
    assert collection.get(where={"wing": "w"}) == {"kind": "get"}
    collection.delete(ids=["1"])
    assert collection.count() == 7

    assert fake.calls == [
        ("add", {"documents": ["d"], "ids": ["1"], "metadatas": [{"wing": "w"}]}),
        ("upsert", {"documents": ["u"], "ids": ["2"], "metadatas": [{"room": "r"}]}),
        ("query", {"query_texts": ["q"]}),
        ("get", {"where": {"wing": "w"}}),
        ("delete", {"ids": ["1"]}),
        ("count", {}),
    ]


def test_chroma_backend_create_false_raises_without_creating_directory(tmp_path):
    palace_path = tmp_path / "missing-palace"

    with pytest.raises(FileNotFoundError):
        ChromaBackend().get_collection(
            str(palace_path),
            collection_name="mempalace_drawers",
            create=False,
        )

    assert not palace_path.exists()


def test_chroma_backend_create_true_creates_directory_and_collection(tmp_path):
    palace_path = tmp_path / "palace"

    collection = ChromaBackend().get_collection(
        str(palace_path),
        collection_name="mempalace_drawers",
        create=True,
    )

    assert palace_path.is_dir()
    assert isinstance(collection, ChromaCollection)

    client = chromadb.PersistentClient(path=str(palace_path))
    client.get_collection("mempalace_drawers")


def test_chroma_backend_creates_collection_with_cosine_distance(tmp_path):
    palace_path = tmp_path / "palace"

    ChromaBackend().get_collection(
        str(palace_path),
        collection_name="mempalace_drawers",
        create=True,
    )

    client = chromadb.PersistentClient(path=str(palace_path))
    col = client.get_collection("mempalace_drawers")
    assert col.metadata.get("hnsw:space") == "cosine"


def test_fix_blob_seq_ids_converts_blobs_to_integers(tmp_path):
    """Simulate a ChromaDB 0.6.x database with BLOB seq_ids and verify repair."""
    db_path = tmp_path / "chroma.sqlite3"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE embeddings (rowid INTEGER PRIMARY KEY, seq_id)")
    conn.execute("CREATE TABLE max_seq_id (rowid INTEGER PRIMARY KEY, seq_id)")
    # Insert BLOB seq_ids like ChromaDB 0.6.x would
    blob_42 = (42).to_bytes(8, byteorder="big")
    blob_99 = (99).to_bytes(8, byteorder="big")
    conn.execute("INSERT INTO embeddings (seq_id) VALUES (?)", (blob_42,))
    conn.execute("INSERT INTO max_seq_id (seq_id) VALUES (?)", (blob_99,))
    conn.commit()
    conn.close()

    _fix_blob_seq_ids(str(tmp_path))

    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT seq_id, typeof(seq_id) FROM embeddings").fetchone()
    assert row == (42, "integer")
    row = conn.execute("SELECT seq_id, typeof(seq_id) FROM max_seq_id").fetchone()
    assert row == (99, "integer")
    conn.close()


def test_fix_blob_seq_ids_noop_without_blobs(tmp_path):
    """No error when seq_ids are already integers."""
    db_path = tmp_path / "chroma.sqlite3"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE embeddings (rowid INTEGER PRIMARY KEY, seq_id INTEGER)")
    conn.execute("INSERT INTO embeddings (seq_id) VALUES (42)")
    conn.commit()
    conn.close()

    _fix_blob_seq_ids(str(tmp_path))

    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT seq_id, typeof(seq_id) FROM embeddings").fetchone()
    assert row == (42, "integer")
    conn.close()


def test_fix_blob_seq_ids_noop_without_database(tmp_path):
    """No error when palace has no chroma.sqlite3."""
    _fix_blob_seq_ids(str(tmp_path))  # should not raise


# ── quarantine_stale_hnsw ─────────────────────────────────────────────────


def _make_palace_with_segment(tmp_path, hnsw_mtime, sqlite_mtime):
    """Helper: build a palace dir with one HNSW segment + sqlite at given mtimes."""
    palace = tmp_path / "palace"
    palace.mkdir()
    (palace / "chroma.sqlite3").write_text("")
    seg = palace / "abcd-1234-5678"
    seg.mkdir()
    (seg / "data_level0.bin").write_text("")
    os.utime(seg / "data_level0.bin", (hnsw_mtime, hnsw_mtime))
    os.utime(palace / "chroma.sqlite3", (sqlite_mtime, sqlite_mtime))
    return palace, seg


def test_quarantine_stale_hnsw_renames_drifted_segment(tmp_path):
    """Segment whose data_level0.bin is 2h older than sqlite gets renamed."""
    now = 1_700_000_000.0
    palace, seg = _make_palace_with_segment(tmp_path, hnsw_mtime=now - 7200, sqlite_mtime=now)
    moved = quarantine_stale_hnsw(str(palace), stale_seconds=3600.0)
    assert len(moved) == 1
    assert ".drift-" in moved[0]
    assert not seg.exists()
    renamed = list(palace.iterdir())
    drift_dirs = [p for p in renamed if ".drift-" in p.name]
    assert len(drift_dirs) == 1
    assert (drift_dirs[0] / "data_level0.bin").exists()


def test_quarantine_stale_hnsw_leaves_fresh_segment_alone(tmp_path):
    """Segment with recent mtime vs sqlite is not touched."""
    now = 1_700_000_000.0
    palace, seg = _make_palace_with_segment(tmp_path, hnsw_mtime=now - 10, sqlite_mtime=now)
    moved = quarantine_stale_hnsw(str(palace), stale_seconds=3600.0)
    assert moved == []
    assert seg.exists()


def test_quarantine_stale_hnsw_no_palace(tmp_path):
    """Missing palace path or chroma.sqlite3: return [] without raising."""
    assert quarantine_stale_hnsw(str(tmp_path / "missing")) == []
    empty = tmp_path / "empty"
    empty.mkdir()
    assert quarantine_stale_hnsw(str(empty)) == []


def test_quarantine_stale_hnsw_skips_already_quarantined(tmp_path):
    """Directories already named with ``.drift-`` suffix are never re-renamed."""
    now = 1_700_000_000.0
    palace = tmp_path / "palace"
    palace.mkdir()
    (palace / "chroma.sqlite3").write_text("")
    os.utime(palace / "chroma.sqlite3", (now, now))
    drift = palace / "abcd-1234.drift-20260101-000000"
    drift.mkdir()
    (drift / "data_level0.bin").write_text("")
    os.utime(drift / "data_level0.bin", (now - 99999, now - 99999))

    moved = quarantine_stale_hnsw(str(palace), stale_seconds=3600.0)
    assert moved == []
    assert drift.exists()
