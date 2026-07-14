"""Tests for the #1222 HNSW capacity probe and BM25-only fallback.

The probe and fallback never load chromadb's HNSW segment, so all of
these tests synthesize the on-disk shape directly: a chroma.sqlite3 with
the relevant schema rows and an ``index_metadata.pickle`` matching what
chromadb 1.5.x writes (``{"id_to_label": {...}, ...}``).
"""

from __future__ import annotations

import os
import pickle
import sqlite3

import pytest

from mempalace.backends.chroma import (
    _hnsw_element_count,
    _vector_segment_id,
    hnsw_capacity_status,
)
from mempalace.searcher import _bm25_only_via_sqlite


COLLECTION = "mempalace_drawers"


# ── Fixtures ──────────────────────────────────────────────────────────


def _seed_chroma_db(palace: str, sqlite_count: int, segment_id: str) -> None:
    """Create a minimal chroma.sqlite3 with one collection + VECTOR segment.

    Mirrors the columns the probe queries: ``segments``, ``collections``,
    ``embeddings``, ``embedding_metadata``. Schema matches chromadb
    1.5.x; column types are kept loose because we read with COUNT(*) and
    SELECT key, *_value rather than driver-specific casts.
    """
    db_path = os.path.join(palace, "chroma.sqlite3")
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE collections (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL
            );
            CREATE TABLE segments (
                id TEXT PRIMARY KEY,
                collection TEXT NOT NULL,
                scope TEXT NOT NULL
            );
            CREATE TABLE embeddings (
                id INTEGER PRIMARY KEY,
                segment_id TEXT NOT NULL,
                embedding_id TEXT NOT NULL,
                seq_id BLOB NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE embedding_metadata (
                id INTEGER REFERENCES embeddings(id),
                key TEXT NOT NULL,
                string_value TEXT,
                int_value INTEGER,
                float_value REAL,
                bool_value INTEGER,
                PRIMARY KEY (id, key)
            );
            CREATE VIRTUAL TABLE embedding_fulltext_search
                USING fts5(string_value, tokenize='trigram');
            """
        )
        col_id = "col-test"
        meta_seg = "seg-meta"
        conn.execute("INSERT INTO collections (id, name) VALUES (?, ?)", (col_id, COLLECTION))
        conn.execute(
            "INSERT INTO segments (id, collection, scope) VALUES (?, ?, 'VECTOR')",
            (segment_id, col_id),
        )
        conn.execute(
            "INSERT INTO segments (id, collection, scope) VALUES (?, ?, 'METADATA')",
            (meta_seg, col_id),
        )
        for i in range(sqlite_count):
            conn.execute(
                """INSERT INTO embeddings (id, segment_id, embedding_id, seq_id)
                   VALUES (?, ?, ?, ?)""",
                (i + 1, segment_id, f"d-{i}", b"\x00\x00\x00\x00\x00\x00\x00\x01"),
            )
        conn.commit()
    finally:
        conn.close()


def _write_pickle(palace: str, segment_id: str, hnsw_count: int) -> None:
    """Write an index_metadata.pickle matching chromadb 1.5.x's shape.

    1.5.x ``__reduce_ex__`` serializes the PersistentData instance as a
    plain dict; we replicate that so the safe unpickler in
    ``_hnsw_element_count`` reads the same bytes shape it would in
    production.
    """
    seg_dir = os.path.join(palace, segment_id)
    os.makedirs(seg_dir, exist_ok=True)
    pickle_path = os.path.join(seg_dir, "index_metadata.pickle")
    state = {
        "dimensionality": 384,
        "total_elements_added": hnsw_count,
        "max_seq_id": None,
        "id_to_label": {f"d-{i}": i for i in range(hnsw_count)},
        "label_to_id": {i: f"d-{i}" for i in range(hnsw_count)},
        "id_to_seq_id": {},
    }
    with open(pickle_path, "wb") as f:
        pickle.dump(state, f, pickle.HIGHEST_PROTOCOL)


# ── _vector_segment_id ────────────────────────────────────────────────


def test_vector_segment_id_returns_uuid(tmp_path):
    seg = "11111111-2222-3333-4444-555555555555"
    _seed_chroma_db(str(tmp_path), sqlite_count=10, segment_id=seg)
    assert _vector_segment_id(str(tmp_path), COLLECTION) == seg


def test_vector_segment_id_no_palace(tmp_path):
    assert _vector_segment_id(str(tmp_path), COLLECTION) is None


def test_vector_segment_id_unknown_collection(tmp_path):
    seg = "11111111-2222-3333-4444-555555555555"
    _seed_chroma_db(str(tmp_path), sqlite_count=10, segment_id=seg)
    assert _vector_segment_id(str(tmp_path), "nope") is None


# ── _hnsw_element_count ───────────────────────────────────────────────


def test_hnsw_element_count_reads_pickle(tmp_path):
    seg = "seg-001"
    _seed_chroma_db(str(tmp_path), sqlite_count=100, segment_id=seg)
    _write_pickle(str(tmp_path), seg, hnsw_count=42)
    assert _hnsw_element_count(str(tmp_path), seg) == 42


def test_hnsw_element_count_missing_pickle(tmp_path):
    seg = "seg-001"
    _seed_chroma_db(str(tmp_path), sqlite_count=100, segment_id=seg)
    # Segment dir doesn't even exist — no flush ever happened.
    assert _hnsw_element_count(str(tmp_path), seg) is None


def test_hnsw_element_count_rejects_arbitrary_class(tmp_path):
    """Pickled references to unallowed classes must not deserialize.

    Guards against a tampered ``index_metadata.pickle`` triggering code
    execution. The unpickler allowlist is the only protection between
    the file and arbitrary import-time side effects. We hand-craft the
    pickle bytes (rather than ``pickle.dump`` a local class) because
    pickle can't serialize locally-defined classes — but the bytes form
    that names an arbitrary stdlib class is a faithful proxy for the
    tampered-file threat we want to test.
    """
    import pickle as _pickle

    seg = "seg-evil"
    seg_dir = tmp_path / seg
    seg_dir.mkdir()
    pickle_path = seg_dir / "index_metadata.pickle"
    # GLOBAL opcode pointing at os.system, then STOP. If the unpickler
    # didn't enforce the allowlist, find_class would resolve os.system
    # and pickle would set up the call. The allowlist must reject it
    # before find_class returns anything.
    payload = b"c" + b"os\nsystem\n" + _pickle.STOP
    pickle_path.write_bytes(payload)
    assert _hnsw_element_count(str(tmp_path), seg) is None


# ── hnsw_capacity_status ──────────────────────────────────────────────


def test_capacity_status_ok_when_balanced(tmp_path):
    seg = "seg-001"
    _seed_chroma_db(str(tmp_path), sqlite_count=1000, segment_id=seg)
    _write_pickle(str(tmp_path), seg, hnsw_count=950)
    info = hnsw_capacity_status(str(tmp_path), COLLECTION)
    assert info["status"] == "ok"
    assert info["diverged"] is False
    assert info["sqlite_count"] == 1000
    assert info["hnsw_count"] == 950


def test_capacity_status_flags_severe_divergence(tmp_path):
    """Reproduces #1222: sqlite has 192k, HNSW frozen at ~16k."""
    seg = "seg-1222"
    _seed_chroma_db(str(tmp_path), sqlite_count=20_000, segment_id=seg)
    _write_pickle(str(tmp_path), seg, hnsw_count=2_000)
    info = hnsw_capacity_status(str(tmp_path), COLLECTION)
    assert info["status"] == "diverged"
    assert info["diverged"] is True
    assert info["divergence"] == 18_000
    assert "repair" in info["message"].lower()


def test_capacity_status_tolerates_flush_lag(tmp_path):
    """A few hundred entries behind sqlite is normal post-mine state."""
    seg = "seg-lag"
    _seed_chroma_db(str(tmp_path), sqlite_count=5_000, segment_id=seg)
    _write_pickle(str(tmp_path), seg, hnsw_count=4_500)
    info = hnsw_capacity_status(str(tmp_path), COLLECTION)
    assert info["diverged"] is False
    assert info["status"] == "ok"


def test_capacity_status_flags_unflushed_with_large_sqlite(tmp_path):
    """No pickle + no on-disk segment + many sqlite rows is real divergence.

    This is the #1222 case: the segment has never written anything to
    disk, sqlite has substantial content, vector search returns nothing
    until rebuild. The capacity probe must flag it so callers fall back
    to BM25 or run repair.
    """
    seg = "seg-noflush"
    _seed_chroma_db(str(tmp_path), sqlite_count=10_000, segment_id=seg)
    # Deliberately do NOT create the segment dir — this is the
    # "never wrote anything" case, distinct from "wrote binary but
    # not pickle" covered below.
    info = hnsw_capacity_status(str(tmp_path), COLLECTION)
    assert info["diverged"] is True
    assert info["hnsw_count"] is None
    assert "never flushed" in info["message"]


def _materialize_segment(
    palace: str, segment_id: str, data_bytes: int, length_elements: int | None
) -> str:
    """Create a segment dir with data_level0.bin and optionally length.bin."""
    seg_dir = os.path.join(palace, segment_id)
    os.makedirs(seg_dir, exist_ok=True)
    with open(os.path.join(seg_dir, "data_level0.bin"), "wb") as f:
        f.write(b"\x00" * data_bytes)
    if length_elements is not None:
        with open(os.path.join(seg_dir, "length.bin"), "wb") as f:
            f.write(b"\x00" * (4 * length_elements))
    return seg_dir


def test_capacity_status_pickle_absent_estimates_from_length_bin(tmp_path):
    """Pickle absent + populated binary: estimate the count from length.bin.

    The old heuristic trusted ``data_level0.bin``'s mere presence and
    reported sqlite_count as the HNSW count. Disproven 2026-07-14:
    chromadb 1.5.7 recreates an *empty* index over a pickle-less segment
    (the "delete the corrupt pickle" #1103 workaround destroyed a
    182k-vector index), and the heuristic reported the emptied segment
    as a healthy full one. length.bin stores 4 bytes per added element,
    so its size is a real count signal that survives the pickle's
    absence.
    """
    seg = "seg-pickle-deleted"
    _seed_chroma_db(str(tmp_path), sqlite_count=132_275, segment_id=seg)
    _materialize_segment(str(tmp_path), seg, data_bytes=200_000, length_elements=132_275)

    info = hnsw_capacity_status(str(tmp_path), COLLECTION)
    assert info["status"] == "ok", info
    assert info["diverged"] is False
    assert info["hnsw_count"] == 132_275
    assert info["divergence"] == 0
    assert "length.bin" in info["message"]


def test_capacity_status_flags_empty_replacement_index(tmp_path):
    """The 2026-07-14 incident shape: a fresh empty index recreated over a
    formerly-full segment (167 KB binary, 100-slot length.bin) while
    sqlite still holds the full corpus. Must report DIVERGED, not the
    false OK the presence-based heuristic produced.
    """
    seg = "seg-emptied"
    _seed_chroma_db(str(tmp_path), sqlite_count=10_000, segment_id=seg)
    _materialize_segment(str(tmp_path), seg, data_bytes=167_600, length_elements=100)

    info = hnsw_capacity_status(str(tmp_path), COLLECTION)
    assert info["status"] == "diverged", info
    assert info["diverged"] is True
    assert info["hnsw_count"] == 100
    assert "repair" in info["message"].lower()


def test_capacity_status_pickle_absent_no_length_bin_is_unverifiable(tmp_path):
    """Binary present but neither pickle nor length.bin: the count cannot
    be verified. Past the absolute threshold, assume the worst and flag
    diverged rather than reporting a fabricated OK.
    """
    seg = "seg-unverifiable"
    _seed_chroma_db(str(tmp_path), sqlite_count=10_000, segment_id=seg)
    _materialize_segment(str(tmp_path), seg, data_bytes=200_000, length_elements=None)

    info = hnsw_capacity_status(str(tmp_path), COLLECTION)
    assert info["diverged"] is True
    assert "unverifiable" in info["message"]


def test_dimensionality_none_pickle_is_normal_on_157(tmp_path):
    """``dimensionality: None`` in the pickle is chromadb 1.5.7's normal
    flush output, NOT a corruption signal — verified 2026-07-14 against
    a pickle chroma itself wrote for a healthy collection. Guard against
    reintroducing a detector keyed on it (the April #1103 diagnosis was
    version-specific); the probe must judge such a segment purely on
    element counts.
    """
    seg = "seg-dim-none"
    _seed_chroma_db(str(tmp_path), sqlite_count=1_000, segment_id=seg)
    pickle_path = os.path.join(str(tmp_path), seg)
    os.makedirs(pickle_path, exist_ok=True)
    state = {
        "dimensionality": None,
        "total_elements_added": 995,
        "max_seq_id": None,
        "id_to_label": {f"d-{i}": i for i in range(995)},
        "label_to_id": {i: f"d-{i}" for i in range(995)},
        "id_to_seq_id": {},
    }
    with open(os.path.join(pickle_path, "index_metadata.pickle"), "wb") as f:
        pickle.dump(state, f, pickle.HIGHEST_PROTOCOL)

    info = hnsw_capacity_status(str(tmp_path), COLLECTION)
    assert info["status"] == "ok", info
    assert info["diverged"] is False
    assert info["hnsw_count"] == 995


def test_auto_drain_fires_on_clean_stranding(tmp_path, monkeypatch):
    """Control: same divergence band with a healthy pickle still drains."""
    from mempalace.backends.chroma import _maybe_auto_drain_unflushed_queue
    import mempalace.repair as repair_mod

    seg = "seg-clean-drain"
    _seed_chroma_db(str(tmp_path), sqlite_count=6_000, segment_id=seg)
    _write_pickle(str(tmp_path), seg, hnsw_count=1_000)

    calls: list[str] = []
    monkeypatch.setattr(
        repair_mod, "recover_unflushed_buffer", lambda *a, **k: calls.append("fired")
    )
    _maybe_auto_drain_unflushed_queue(str(tmp_path))
    assert calls == ["fired"]


def test_capacity_status_pickle_absent_with_empty_data_file_still_skips(tmp_path):
    """A 0-byte ``data_level0.bin`` is equivalent to no data file —
    don't fall into the "loadable" branch just because the file
    exists; the segment really hasn't written anything yet.
    """
    seg = "seg-empty-data"
    _seed_chroma_db(str(tmp_path), sqlite_count=500, segment_id=seg)
    seg_dir = os.path.join(str(tmp_path), seg)
    os.makedirs(seg_dir, exist_ok=True)
    open(os.path.join(seg_dir, "data_level0.bin"), "wb").close()  # 0 bytes

    info = hnsw_capacity_status(str(tmp_path), COLLECTION)
    # Under the divergence threshold and no real binary on disk →
    # quiet "not yet flushed" status, NOT a false-positive OK.
    assert info["diverged"] is False
    assert info["status"] != "ok"  # specifically: should be "unknown"
    assert info["hnsw_count"] is None


def test_capacity_status_quiet_for_empty_palace(tmp_path):
    info = hnsw_capacity_status(str(tmp_path), COLLECTION)
    assert info["diverged"] is False
    assert info["status"] == "unknown"


# ── BM25-only sqlite fallback ─────────────────────────────────────────


def _seed_drawers(palace: str, segment_id: str, drawers: list[tuple[str, dict, str]]) -> None:
    """Insert (text, metadata, embedding_id) tuples into a seeded palace.

    Replaces the bare ``embeddings`` rows from ``_seed_chroma_db`` so the
    sqlite count matches what we insert here.
    """
    db_path = os.path.join(palace, "chroma.sqlite3")
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DELETE FROM embeddings")
        for i, (text, meta, eid) in enumerate(drawers, start=1):
            conn.execute(
                """INSERT INTO embeddings (id, segment_id, embedding_id, seq_id)
                   VALUES (?, ?, ?, ?)""",
                (i, segment_id, eid, b"\x00" * 8),
            )
            conn.execute(
                """INSERT INTO embedding_metadata (id, key, string_value)
                   VALUES (?, 'chroma:document', ?)""",
                (i, text),
            )
            conn.execute(
                "INSERT INTO embedding_fulltext_search (rowid, string_value) VALUES (?, ?)",
                (i, text),
            )
            for k, v in meta.items():
                if isinstance(v, int):
                    conn.execute(
                        """INSERT INTO embedding_metadata (id, key, int_value)
                           VALUES (?, ?, ?)""",
                        (i, k, v),
                    )
                else:
                    conn.execute(
                        """INSERT INTO embedding_metadata (id, key, string_value)
                           VALUES (?, ?, ?)""",
                        (i, k, str(v)),
                    )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def palace_with_drawers(tmp_path):
    seg = "seg-bm25"
    _seed_chroma_db(str(tmp_path), sqlite_count=0, segment_id=seg)
    drawers = [
        (
            "ChromaDB segfault on every tool call after HNSW divergence",
            {"wing": "ops", "room": "incidents", "source_file": "/x/incident.md"},
            "d-1",
        ),
        (
            "Memory palace technique using rooms and drawers for recall",
            {"wing": "design", "room": "metaphor", "source_file": "/x/design.md"},
            "d-2",
        ),
        (
            "Repair rebuild backs up only the sqlite database",
            {"wing": "ops", "room": "runbook", "source_file": "/x/repair.md"},
            "d-3",
        ),
    ]
    _seed_drawers(str(tmp_path), seg, drawers)
    return tmp_path


def test_bm25_fallback_returns_matches(palace_with_drawers):
    out = _bm25_only_via_sqlite("segfault chromadb", str(palace_with_drawers), n_results=5)
    assert out["fallback"] == "bm25_only_via_sqlite"
    assert len(out["results"]) >= 1
    top = out["results"][0]
    # The incident drawer is the closest BM25 match for these terms.
    assert "segfault" in top["text"].lower()
    assert top["matched_via"] == "bm25_sqlite"
    # Vector fields are intentionally absent in fallback mode.
    assert top["similarity"] is None
    assert top["distance"] is None


def test_bm25_fallback_filters_by_wing(palace_with_drawers):
    out = _bm25_only_via_sqlite(
        "memory palace recall", str(palace_with_drawers), wing="design", n_results=5
    )
    assert all(r["wing"] == "design" for r in out["results"])


def test_bm25_fallback_no_palace(tmp_path):
    out = _bm25_only_via_sqlite("anything", str(tmp_path))
    assert "error" in out


def test_bm25_fallback_handles_short_query(palace_with_drawers):
    """Single-character tokens are unmatchable in trigram FTS5 — must
    not crash, must fall back to the recency window."""
    out = _bm25_only_via_sqlite("a", str(palace_with_drawers), n_results=5)
    # Falls back to recency window; returns whatever it can rank.
    assert out["fallback"] == "bm25_only_via_sqlite"
    assert isinstance(out["results"], list)


# ── repair.status CLI command ─────────────────────────────────────────


def test_repair_status_reports_diverged(tmp_path, capsys):
    """The status command prints DIVERGED and recommends rebuild."""
    from mempalace.repair import status as repair_status

    seg = "seg-status"
    _seed_chroma_db(str(tmp_path), sqlite_count=20_000, segment_id=seg)
    _write_pickle(str(tmp_path), seg, hnsw_count=2_000)
    out = repair_status(palace_path=str(tmp_path))
    captured = capsys.readouterr().out
    assert "DIVERGED" in captured
    assert "mempalace repair`" in captured
    assert out["drawers"]["diverged"] is True


def test_repair_status_quiet_on_healthy_palace(tmp_path, capsys):
    from mempalace.repair import status as repair_status

    seg = "seg-status-ok"
    _seed_chroma_db(str(tmp_path), sqlite_count=500, segment_id=seg)
    _write_pickle(str(tmp_path), seg, hnsw_count=480)
    repair_status(palace_path=str(tmp_path))
    captured = capsys.readouterr().out
    assert "DIVERGED" not in captured
    assert "Recommended" not in captured


# ── tool_status sqlite fallback (#1222 short-circuit) ─────────────────


def test_tool_status_via_sqlite_returns_breakdown(palace_with_drawers, monkeypatch):
    """When _vector_disabled is set, tool_status reads counts from sqlite
    instead of opening a chromadb client."""
    from mempalace import mcp_server

    # _config.palace_path is a read-only property; swap the whole object
    # for a tiny stand-in so we don't have to monkey with the real
    # MempalaceConfig.
    class _Cfg:
        palace_path = str(palace_with_drawers)

    monkeypatch.setattr(mcp_server, "_config", _Cfg())
    monkeypatch.setattr(mcp_server, "_vector_disabled", True)
    monkeypatch.setattr(mcp_server, "_vector_disabled_reason", "test divergence")

    out = mcp_server._tool_status_via_sqlite()
    assert out["vector_disabled"] is True
    assert out["vector_disabled_reason"] == "test divergence"
    assert out["total_drawers"] == 3
    # Wing breakdown comes from the seeded palace_with_drawers fixture:
    # ops×2 (incident + repair runbook), design×1 (metaphor).
    assert out["wings"].get("ops") == 2
    assert out["wings"].get("design") == 1
