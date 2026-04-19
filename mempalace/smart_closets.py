"""smart_closets.py — Deterministic, no-LLM closet line builder.

Three layers stacked on top of the regex extractor in `palace.build_closet_lines`:

1. **Corpus-wide TF-IDF over n-grams.** Counts 1-3-gram document frequencies
   across every drawer in `mempalace_drawers`, persists the IDF table to
   ``<palace>/smart_closets_idf.json`` so subsequent mines reuse it. The
   IDF formula is the same Lucene/BM25+ smoothed form used by
   ``searcher.py:_bm25_scores`` — single source of truth for the ranking
   math, no parallel implementation.

2. **N-gram candidates with stopword/punctuation filtering.** Replaces the
   single-capitalized-word regex with 1-3-token phrases that don't start
   or end on a stopword, so multi-word concepts ("session refresh
   strategy", "cursor mine", "/clear data loss") survive into the index.

3. **Embedder rerank with MMR.** Embeds the document once and each top-K
   TF-IDF candidate via ChromaDB's already-loaded
   ``DefaultEmbeddingFunction``. Reranks by cosine similarity to the
   document and applies maximal-marginal-relevance to suppress
   near-duplicate candidates. Result: phrase ranking semantically aligned
   with the document, fully deterministic given the fixed ONNX model
   weights.

Output: a list of closet pointer lines in the same ``topic|entities|→drawers``
format consumed by ``palace.upsert_closet_lines`` — drop-in replacement
for ``palace.build_closet_lines`` when an embedder is available.

Trade-offs vs the regex baseline:
  - Catches multi-word concepts and lowercase technical terms regex misses.
  - IDF replaces the hardcoded stoplist with a corpus-learned one.
  - MMR-diversified top-K avoids 8 near-identical "auth", "auth flow",
    "auth flow architecture" entries that swamp a small closet budget.

Trade-offs vs an LLM:
  - No quote attribution or summary sentence (regex/LLM fields not produced
    by this module — quotes are a separate, deterministic primitive in
    ``extract_quotes``).
  - Phrasing comes verbatim from the document; an LLM could restate a
    concept in better-embedding-friendly language.

This module never calls a network endpoint. The only model used is the
local ONNX embedder ChromaDB has already loaded for drawer storage.
"""

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Optional, Sequence

import numpy as np

# Tokenizer: words are runs of word-chars optionally containing internal
# hyphens, apostrophes, or underscores. Slash is intentionally NOT a
# token character — file paths get split into their components instead
# of producing noisy fragments like "users/peterwang/code/playground".
# Lowercased on emission so case never affects n-gram identity.
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'\-]{0,40}")
_MIN_TOKEN_LEN = 3
# Cheap noise filter: 2-3-char unit-like fragments that survive everywhere
# (css px/em/rem, time ms/sec, mem kb/mb/gb). These are real words but
# wash out IDF without contributing meaning to a closet pointer.
_NOISE_TOKENS = frozenset(
    {
        "px", "em", "rem", "vh", "vw", "ms", "sec", "kb", "mb", "gb", "tb",
        "kbs", "mbs", "fps", "rgb", "rgba", "hex", "url", "src", "alt",
        "tsx", "jsx", "css", "ts", "js", "py", "rb", "md", "txt", "log",
        "tmp", "var", "let", "fn", "fns", "obj", "objs", "str", "strs",
        "int", "ints", "num", "nums", "bool", "bools", "args", "kwargs",
        "ctx", "cb", "cbs", "btn", "btns", "img", "imgs", "div", "divs",
        "id", "ids", "ref", "refs", "key", "keys", "val", "vals",
    }
)

# Compact English stopword list. Deliberately small: aggressive stopwording
# erases real terms (e.g. "for" in "for-loop"). Items here are *only* the
# tokens that appear so often everywhere they wash out IDF entirely.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "else", "then", "of",
        "to", "in", "on", "at", "by", "for", "with", "from", "as", "is",
        "are", "was", "were", "be", "been", "being", "do", "does", "did",
        "have", "has", "had", "having", "i", "you", "we", "they", "he",
        "she", "it", "this", "that", "these", "those", "my", "your",
        "our", "their", "his", "her", "its", "me", "us", "them", "him",
        "what", "when", "where", "why", "how", "who", "which", "whom",
        "not", "no", "yes", "so", "just", "also", "very", "really", "can",
        "will", "would", "could", "should", "may", "might", "must", "any",
        "some", "all", "more", "most", "less", "few", "many", "much",
        "than", "too", "into", "out", "over", "under", "up", "down",
        "now", "here", "there", "yet", "still", "ok", "okay", "well",
        "like", "get", "got", "go", "going", "come", "let", "make",
        "made", "want", "need", "see", "say", "said", "tell", "told",
        "know", "think", "use", "used", "using", "via", "thing", "stuff",
    }
)

# IDF index file lives next to chroma.sqlite3 inside the palace dir. Versioned
# so a tokenization or stopword change forces a rebuild.
IDF_INDEX_FILENAME = "smart_closets_idf.json"
IDF_INDEX_VERSION = 1

# How many TF-IDF candidates to embed before MMR. Higher = better recall at
# the cost of more embedder calls per source file.
DEFAULT_CANDIDATE_POOL = 80
# How many phrases to actually emit per source file.
DEFAULT_TOP_K = 12
# MMR diversity weight. λ=1.0 → pure relevance, λ=0.0 → pure diversity.
DEFAULT_MMR_LAMBDA = 0.65


def tokenize(text: str) -> list:
    """Lowercased word tokens. Tokens shorter than ``_MIN_TOKEN_LEN`` and
    known noise tokens are dropped at this stage so they can't enter
    n-grams at all. Bounded length per token to prevent runaway base64
    captures from dominating IDF counts.
    """
    out = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0).lower()
        if len(tok) < _MIN_TOKEN_LEN:
            continue
        if tok in _NOISE_TOKENS:
            continue
        out.append(tok)
    return out


def _is_stopword(token: str) -> bool:
    return token in _STOPWORDS


def extract_ngrams(tokens: Sequence[str], n_min: int = 1, n_max: int = 3) -> list:
    """Slide a 1..n_max window over tokens. Reject n-grams that:

      - start or end on a stopword (multi-word phrases shouldn't anchor on "the")
      - consist entirely of stopwords (single-token n=1 stopwords excluded)

    The minimum length and noise filters are already applied in ``tokenize``,
    so n-grams here are clean by construction. Returns a deterministic,
    ordered list (duplicates kept; counted later).
    """
    out = []
    n_tokens = len(tokens)
    for n in range(n_min, n_max + 1):
        if n_tokens < n:
            continue
        for i in range(n_tokens - n + 1):
            gram = tokens[i : i + n]
            if all(_is_stopword(t) for t in gram):
                continue
            if _is_stopword(gram[0]) or _is_stopword(gram[-1]):
                continue
            out.append(" ".join(gram))
    return out


# --------------------------------------------------------------------------
# IDF index — one pass over all drawers, persisted to disk
# --------------------------------------------------------------------------


def _idf_index_path(palace_path: str) -> str:
    return os.path.join(palace_path, IDF_INDEX_FILENAME)


def load_idf_index(palace_path: str) -> Optional[dict]:
    """Return the persisted IDF index, or None if missing/stale/corrupt."""
    path = _idf_index_path(palace_path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("version") != IDF_INDEX_VERSION:
        return None
    if not isinstance(data.get("idf"), dict):
        return None
    if not isinstance(data.get("n_docs"), int):
        return None
    return data


def save_idf_index(palace_path: str, n_docs: int, idf: dict) -> None:
    path = _idf_index_path(palace_path)
    tmp = path + ".tmp"
    payload = {"version": IDF_INDEX_VERSION, "n_docs": n_docs, "idf": idf}
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def build_idf_index(
    drawers_col,
    n_max: int = 3,
    min_df: int = 2,
    progress: bool = False,
) -> dict:
    """Walk every drawer and compute document-frequency for each n-gram,
    then convert to the same smoothed IDF formula ``searcher.py:_bm25_scores``
    uses (Lucene / BM25+ form): ``log((N - df + 0.5) / (df + 0.5) + 1)``.

    n-grams seen in fewer than ``min_df`` documents are dropped before
    persisting — they have huge IDF but are too rare to act as useful
    closet pointers (typo-of-the-day, accidentally captured base64).

    Returns ``{"version": 1, "n_docs": N, "idf": {ngram: idf_score}}``.
    """
    BATCH = 500
    df: dict = defaultdict(int)
    n_docs = 0
    offset = 0
    while True:
        batch = drawers_col.get(
            include=["documents"],
            limit=BATCH,
            offset=offset,
        )
        docs = batch.get("documents", []) or []
        if not docs:
            break
        for doc in docs:
            n_docs += 1
            tokens = tokenize(doc)
            seen_in_doc = set(extract_ngrams(tokens, 1, n_max))
            for gram in seen_in_doc:
                df[gram] += 1
        offset += len(docs)
        if progress:
            print(f"  IDF: {n_docs} drawers scanned, {len(df)} unique n-grams")
        if len(docs) < BATCH:
            break

    if n_docs == 0:
        return {"version": IDF_INDEX_VERSION, "n_docs": 0, "idf": {}}

    idf = {}
    for gram, freq in df.items():
        if freq < min_df:
            continue
        idf[gram] = math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1)
    return {"version": IDF_INDEX_VERSION, "n_docs": n_docs, "idf": idf}


# --------------------------------------------------------------------------
# Per-document scoring
# --------------------------------------------------------------------------


def score_candidates_tfidf(
    content: str,
    idf: dict,
    n_max: int = 3,
    pool_size: int = DEFAULT_CANDIDATE_POOL,
    fallback_idf: float = 1.0,
) -> list:
    """Score every n-gram in ``content`` by ``tf × idf`` and return the top
    ``pool_size`` (phrase, score) pairs sorted by score descending.

    n-grams not in the IDF index get a small ``fallback_idf`` so brand-new
    distinctive phrases (rare globally, naturally) still surface — without
    swamping the ranking when they're noisy.
    """
    tokens = tokenize(content)
    grams = extract_ngrams(tokens, 1, n_max)
    if not grams:
        return []
    tf = Counter(grams)
    scored = []
    for gram, freq in tf.items():
        weight = idf.get(gram, fallback_idf)
        if weight <= 0:
            continue
        scored.append((gram, freq * weight))
    # Deterministic tie-breaking: by (-score, gram) so identical scores
    # always produce the same ranking across runs.
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored[:pool_size]


# --------------------------------------------------------------------------
# Embedder rerank + MMR
# --------------------------------------------------------------------------


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return matrix / norms


def _embed(embedder, texts: Sequence[str]) -> np.ndarray:
    """Call the chroma embedder, return a normalized (n, d) ndarray.
    Splits into reasonable batches so the embedder isn't asked to chew on
    a huge candidate pool in one shot.
    """
    BATCH = 64
    out = []
    for i in range(0, len(texts), BATCH):
        chunk = list(texts[i : i + BATCH])
        vecs = embedder(chunk)
        out.extend(vecs)
    arr = np.asarray(out, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return _normalize_rows(arr)


def mmr_select(
    candidates: Sequence[str],
    candidate_vecs: np.ndarray,
    query_vec: np.ndarray,
    top_k: int,
    lambda_: float = DEFAULT_MMR_LAMBDA,
) -> list:
    """Maximal Marginal Relevance — pick top_k that balance high relevance
    to the query and low similarity to already-picked items.

    Vectors must be unit-norm. Returns a list of candidate indices.
    """
    if not len(candidates):
        return []
    n = len(candidates)
    relevance = candidate_vecs @ query_vec
    selected: list = []
    remaining = list(range(n))
    while remaining and len(selected) < top_k:
        if not selected:
            best = max(remaining, key=lambda i: (relevance[i], -i))
            selected.append(best)
            remaining.remove(best)
            continue
        sel_vecs = candidate_vecs[selected]
        # max similarity of each remaining cand to any already-selected cand
        max_sim = (candidate_vecs[remaining] @ sel_vecs.T).max(axis=1)
        scores = lambda_ * relevance[remaining] - (1 - lambda_) * max_sim
        # Deterministic tie-break on lower index
        best_local = int(np.lexsort((remaining, -scores))[0])
        best_global = remaining[best_local]
        selected.append(best_global)
        remaining.pop(best_local)
    return selected


# --------------------------------------------------------------------------
# Quote extraction (transcript-aware deterministic primitive)
# --------------------------------------------------------------------------


_USER_LINE_RE = re.compile(r"^(?:>\s+|\[user\]\s+|\[USER\]\s+)(.+?)$", re.MULTILINE)


def extract_quotes(content: str, idf: dict, top_k: int = 3, max_chars: int = 220) -> list:
    """Pull user-marked lines from a normalized transcript and rank them by
    cumulative IDF (distinctive vocabulary = informative quote).
    No-op for non-transcript content (no markers → empty list).
    """
    matches = _USER_LINE_RE.findall(content)
    if not matches:
        return []
    scored = []
    for line in matches:
        line = line.strip()
        if len(line) < 12 or len(line) > 600:
            continue
        toks = tokenize(line)
        score = sum(idf.get(t, 0.0) for t in toks)
        if score <= 0:
            continue
        scored.append((score, line))
    scored.sort(key=lambda p: (-p[0], p[1]))
    out = []
    for _, line in scored[:top_k]:
        out.append(line[:max_chars])
    return out


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def build_smart_closet_lines(
    source_file: str,
    drawer_ids: Sequence[str],
    content: str,
    wing: str,
    room: str,
    *,
    idf_index: dict,
    embedder,
    top_k: int = DEFAULT_TOP_K,
    pool_size: int = DEFAULT_CANDIDATE_POOL,
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
    quote_top_k: int = 3,
) -> list:
    """Drop-in upgrade for ``palace.build_closet_lines``.

    Pipeline: TF-IDF candidate pool → embedder rerank w/ MMR → top-k
    phrases → emit ``topic|entities|→drawers`` lines, prepended with
    extracted user-quote lines for transcript content.

    Falls back to an empty list if anything upstream is missing
    (no drawers, empty content, no IDF index). The caller is expected to
    fall back to the regex builder in that case.
    """
    if not drawer_ids or not content:
        return []
    idf = idf_index.get("idf", {}) if isinstance(idf_index, dict) else {}
    if not idf:
        return []

    drawer_ref = ",".join(drawer_ids[:3])
    entities_str = ""  # smart pipeline derives entities from candidates themselves

    candidates = score_candidates_tfidf(content, idf, pool_size=pool_size)
    if not candidates:
        return []
    cand_phrases = [c for c, _ in candidates]

    try:
        cand_vecs = _embed(embedder, cand_phrases)
        doc_vec = _embed(embedder, [content[:8000]])[0]
    except Exception:
        # Embedder failed (model not loaded, OOM, etc) — fall back to pure TF-IDF order
        picked = list(range(min(top_k, len(cand_phrases))))
    else:
        picked = mmr_select(cand_phrases, cand_vecs, doc_vec, top_k, lambda_=mmr_lambda)

    lines = []
    for line in extract_quotes(content, idf, top_k=quote_top_k):
        lines.append(f"{line}|{entities_str}|→{drawer_ref}")
    for idx in picked:
        phrase = cand_phrases[idx]
        lines.append(f"{phrase}|{entities_str}|→{drawer_ref}")
    return lines


def get_or_build_idf_index(
    palace_path: str, drawers_col, force_rebuild: bool = False, progress: bool = False
) -> dict:
    """Load the persisted IDF index if present and current; otherwise build
    one from the drawer corpus and persist it. Callers on the hot path
    (mine, hooks) should call this once per process, not per file.
    """
    if not force_rebuild:
        cached = load_idf_index(palace_path)
        if cached is not None:
            return cached
    if progress:
        print("  Building corpus-wide IDF index...")
    index = build_idf_index(drawers_col, progress=progress)
    save_idf_index(palace_path, index["n_docs"], index["idf"])
    return index


# Stable signature suffix the smart pipeline tags onto closet metadata so a
# later inspector can tell smart-closets from regex/LLM closets.
SMART_CLOSET_SIGNATURE = (
    f"smart-tfidf-mmr-v{IDF_INDEX_VERSION}-"
    f"{hashlib.sha1((str(DEFAULT_TOP_K) + str(DEFAULT_MMR_LAMBDA)).encode()).hexdigest()[:6]}"
)


__all__ = [
    "tokenize",
    "extract_ngrams",
    "build_idf_index",
    "load_idf_index",
    "save_idf_index",
    "get_or_build_idf_index",
    "score_candidates_tfidf",
    "mmr_select",
    "extract_quotes",
    "build_smart_closet_lines",
    "SMART_CLOSET_SIGNATURE",
]


def _embedder_from_drawers_col(drawers_col):
    """Reach through the ChromaCollection adapter to the inner chroma
    Collection's DefaultEmbeddingFunction. Returns a callable
    ``(texts: list[str]) -> list[list[float]]`` or None if the embedder
    isn't accessible (e.g. a future backend that doesn't expose it).
    """
    inner = getattr(drawers_col, "_collection", None)
    if inner is None:
        return None
    return getattr(inner, "_embedding_function", None)
