"""bridge_utils.py — Model-agnostic helpers for Bridge Head / MH Head discovery.

Covers:
  - Gold answer span extraction from eval.*_pred fields
  - Head index helpers (layer, head, flat)
  - Filtered-record IO and deterministic sampling
  - Wide-format parquet + jsonl dual-save
  - Bridge score computation and z-score normalisation
  - MH language-specificity filtering (general-head deduplication)
  - Subsampling-based stability (no model re-run)
  - Overlap metrics (Jaccard, enrichment, hypergeometric, Spearman)
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

from utils.common import load_json, set_seed

# ---------------------------------------------------------------------------
# Gold extraction
# ---------------------------------------------------------------------------

# Question-marker patterns that signal the start of a follow-up prompt.
# Order matters: longest / most specific first.
_QUESTION_MARKERS: list[str] = [
    "Question:",
    "[Question]",
    "질문:",
    "[질문]",
    "问题",
    "質問",
    "Pregunta:",
    "Pertanyaan:",
    "Câu hỏi:",
]

# Sentence-terminator characters.  The regex stops *before* these characters.
_SENT_TERM_RE = re.compile(r"[.。?？!！\n]")

# Reasonable upper bound on clean answer length (characters).
_MAX_GOLD_LEN = 200


def extract_answer_span(
    pred_text: str,
    mode: str = "first_span",
    tokenizer: Any | None = None,
) -> str | None:
    """Extract a clean answer span from a raw model prediction string.

    Parameters
    ----------
    pred_text:
        Raw model prediction, e.g. ``"Alfonso VIII Question:Who is..."``.
    mode:
        - ``first_span``  — truncate at question marker, then at sentence boundary.
        - ``first_word``  — first whitespace-delimited token.
        - ``first_token`` — first subword token as decoded by *tokenizer* (requires
          ``tokenizer`` kwarg; falls back to ``first_word`` if not provided).
    tokenizer:
        HuggingFace tokenizer; only used when ``mode == "first_token"``.

    Returns
    -------
    str or None
        Cleaned span, or ``None`` if empty / too long (caller should log + skip).
    """
    if not pred_text or not pred_text.strip():
        return None

    text = pred_text.strip()

    if mode == "first_token":
        if tokenizer is not None:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                return None
            span = tokenizer.decode([ids[0]]).strip()
            return span if span else None
        mode = "first_word"  # fallback

    if mode == "first_word":
        span = text.split()[0] if text.split() else ""
        return span if span else None

    # --- first_span (default) ---
    # 1. Truncate at the earliest question marker.
    cut = len(text)
    for marker in _QUESTION_MARKERS:
        idx = text.find(marker)
        if idx != -1 and idx < cut:
            cut = idx
    text = text[:cut]

    # 2. Truncate at the first sentence terminator.
    m = _SENT_TERM_RE.search(text)
    if m:
        text = text[: m.start()]

    text = text.strip()
    if not text or len(text) > _MAX_GOLD_LEN:
        return None
    return text


# ---------------------------------------------------------------------------
# Head indexing helpers
# ---------------------------------------------------------------------------

def head_id(layer: int, head: int) -> str:
    """Return canonical head string, e.g. ``"L12H3"``."""
    return f"L{layer}H{head}"


def flat_index(layer: int, head: int, n_heads: int) -> int:
    """Convert (layer, head) to a flat integer index."""
    return layer * n_heads + head


def head_from_flat(flat: int, n_heads: int) -> tuple[int, int]:
    """Convert flat index back to (layer, head)."""
    return divmod(flat, n_heads)


def all_head_pairs(n_layers: int, n_heads: int) -> list[tuple[int, int]]:
    """Return all (layer, head) pairs in row-major order."""
    return [(l, h) for l in range(n_layers) for h in range(n_heads)]


def head_column_names(n_layers: int, n_heads: int) -> list[str]:
    """Column names for wide score matrices, e.g. ``["L0H0", "L0H1", ...]``."""
    return [head_id(l, h) for l, h in all_head_pairs(n_layers, n_heads)]


# ---------------------------------------------------------------------------
# IO and sampling
# ---------------------------------------------------------------------------

def load_filtered_records(
    input_root: Path,
    model_short: str,
    lang: str,
) -> list[dict[str, Any]]:
    """Load ``correct_{model_short}_{lang}.json`` from the standard filtered path.

    Uses ``load_json`` from ``common.py`` which already unwraps the ``data`` key.
    """
    path = input_root / model_short / "filtered" / lang / f"correct_{model_short}_{lang}.json"
    return load_json(path)


def load_cross_lang_records(
    input_root: Path,
    model_short: str,
    lang_a: str,
    lang_b: str,
) -> list[dict[str, Any]]:
    """Load a pre-built cross-lang both-correct paired file.

    File pattern: ``{input_root}/{model_short}/filtered/cross_lang/
    {lang_a}_{lang_b}_both_correct_{model_short}.json``

    Each record contains a ``langs`` dict keyed by language code, with
    ``prompts`` and ``eval`` sub-dicts for each language.

    Returns the ``data`` list from the JSON.
    """
    fname = f"{lang_a}_{lang_b}_both_correct_{model_short}.json"
    path  = input_root / model_short / "filtered" / "cross_lang" / fname
    return load_json(path)


def flatten_cross_lang_record(rec: dict[str, Any], lang: str) -> dict[str, Any]:
    """Return a single-language flat record compatible with step5 helpers.

    Pulls ``prompts``, ``eval``, and entity labels from ``rec['langs'][lang]``
    and merges top-level metadata (hop_id, id, …) so the result looks like a
    normal per-lang filtered record.
    """
    lang_data = rec["langs"][lang]
    flat = {
        "id":     rec.get("id"),
        "hop_id": rec.get("hop_id"),
        "r1":     rec.get("r1"),
        "r2":     rec.get("r2"),
        "lang":   lang,
        "e2_label_en": lang_data.get("e2_label_en", ""),
        "e2":     {"label": lang_data.get("e2_label", ""),
                   "label_en": lang_data.get("e2_label_en", "")},
        "prompts": lang_data.get("prompts", {}),
        "eval":    lang_data.get("eval", {}),
    }
    return flat


def sample_records(
    records: list[dict[str, Any]],
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Return up to *n* records sampled deterministically without replacement."""
    if n <= 0 or n >= len(records):
        return list(records)
    set_seed(seed)
    return random.sample(records, n)


def _df_to_jsonl(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to newline-delimited JSON (jsonl)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_scores_matrix(
    score_matrix: np.ndarray,
    item_ids: list[str],
    n_layers: int,
    n_heads: int,
    path: Path,
) -> None:
    """Save an item × head score matrix as both ``.parquet`` and ``.jsonl``.

    Parameters
    ----------
    score_matrix:
        Shape ``(n_items, n_layers * n_heads)``.
    item_ids:
        List of record IDs, length ``n_items``.
    n_layers, n_heads:
        Used to build column names.
    path:
        Target path for the ``.parquet`` file; ``.jsonl`` sibling is created
        automatically.
    """
    cols = head_column_names(n_layers, n_heads)
    df = pd.DataFrame(score_matrix, columns=cols)
    df.insert(0, "item_id", item_ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    _df_to_jsonl(df, path.with_suffix(".jsonl"))


def save_head_score_df(df: pd.DataFrame, path: Path) -> None:
    """Save a pre-built per-head summary DataFrame as ``.parquet`` and ``.jsonl``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    _df_to_jsonl(df, path.with_suffix(".jsonl"))


def load_scores_matrix(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load a wide-format score parquet back to ``(matrix, item_ids)``."""
    df = pd.read_parquet(path)
    item_ids = df["item_id"].tolist()
    matrix = df.drop(columns=["item_id"]).values.astype(np.float32)
    return matrix, item_ids


# ---------------------------------------------------------------------------
# Bridge and MH score computation
# ---------------------------------------------------------------------------

def zscores(arr: np.ndarray) -> np.ndarray:
    """Z-normalise *arr* over all elements (head dimension)."""
    mu, sigma = arr.mean(), arr.std()
    if sigma < 1e-12:
        return np.zeros_like(arr)
    return (arr - mu) / sigma


def compute_bridge_scores(
    mean_fh: np.ndarray,
    mean_th: np.ndarray,
    mean_sh: np.ndarray,
) -> np.ndarray:
    """Compute Bridge score: ``z(FH) + z(TH) - z(SH)``.

    All inputs should be 1-D arrays of shape ``(n_heads_total,)``.
    """
    return zscores(mean_fh) + zscores(mean_th) - zscores(mean_sh)


def compute_mh_centric_scores(
    mean_fh: np.ndarray,
    mean_th: np.ndarray,
    mean_sh: np.ndarray,
) -> np.ndarray:
    """Compute MH-centric score: ``z(TH) - z(FH) - z(SH)``.

    Highlights heads important for the two-hop task but NOT for either
    individual hop, capturing purely compositional computation.
    """
    return zscores(mean_th) - zscores(mean_fh) - zscores(mean_sh)


def top_percent_set(scores: np.ndarray, percent: float) -> set[int]:
    """Return flat indices of the top *percent* (0.0–1.0) scoring heads."""
    k = max(1, int(len(scores) * percent))
    return set(np.argsort(scores)[::-1][:k].tolist())


def top_k_set(scores: np.ndarray, k: int) -> set[int]:
    """Return flat indices of the top-*k* scoring heads."""
    k = min(k, len(scores))
    return set(np.argsort(scores)[::-1][:k].tolist())


# ---------------------------------------------------------------------------
# MH language-specificity filtering
# ---------------------------------------------------------------------------

def find_general_heads(
    mh_scores_by_lang: dict[str, np.ndarray],
    top_percent: float,
    min_repetition: int = 3,
) -> set[int]:
    """Identify heads that appear in the top-percent set of ≥ min_repetition languages.

    Parameters
    ----------
    mh_scores_by_lang:
        ``{lang: mean_mh_scores[n_heads_total]}``.
    top_percent:
        Top fraction to consider for each language.
    min_repetition:
        A head is 'general' if it appears in this many or more languages' top set.

    Returns
    -------
    set of flat head indices classified as language-general.
    """
    lang_top_sets: list[set[int]] = [
        top_percent_set(scores, top_percent)
        for scores in mh_scores_by_lang.values()
    ]
    n_heads_total = next(iter(mh_scores_by_lang.values())).shape[0]
    counts = np.zeros(n_heads_total, dtype=np.int32)
    for top_set in lang_top_sets:
        for h in top_set:
            counts[h] += 1
    return {int(h) for h in np.where(counts >= min_repetition)[0]}


def make_specific_set(
    raw_set: set[int],
    general_heads: set[int],
) -> set[int]:
    """Return *raw_set* minus *general_heads* (language-specific heads)."""
    return raw_set - general_heads


# ---------------------------------------------------------------------------
# Subsampling stability
# ---------------------------------------------------------------------------

def _bridge_score_from_rows(
    fh_mat: np.ndarray,
    th_mat: np.ndarray,
    sh_mat: np.ndarray,
    idx: np.ndarray,
    score_fn=None,
) -> np.ndarray:
    if score_fn is None:
        score_fn = compute_bridge_scores
    mean_fh = fh_mat[idx].mean(axis=0)
    mean_th = th_mat[idx].mean(axis=0)
    mean_sh = sh_mat[idx].mean(axis=0)
    return score_fn(mean_fh, mean_th, mean_sh)


def subsampling_stability(
    fh_mat: np.ndarray,
    th_mat: np.ndarray,
    sh_mat: np.ndarray,
    n_sub: int,
    n_bootstrap: int,
    seed: int,
    top_percent: float,
    top_k: int = 0,
    score_fn=None,
) -> dict[str, np.ndarray]:
    """Compute per-head stability via subsampling without replacement.

    Parameters
    ----------
    fh_mat, th_mat, sh_mat:
        Item × head score matrices (rows = items, cols = heads).
    n_sub:
        Number of rows to draw per bootstrap run.
    n_bootstrap:
        Number of bootstrap runs.
    seed:
        RNG seed for reproducibility.
    top_percent, top_k:
        Thresholds for defining the top-head set per run.

    Returns
    -------
    dict with keys ``"stability_top_percent"`` and ``"stability_top_k"``,
    each an array of shape ``(n_heads_total,)`` holding the fraction of
    bootstrap runs in which that head appeared in the respective top set.
    """
    rng = np.random.default_rng(seed)
    n_items = fh_mat.shape[0]
    n_heads = fh_mat.shape[1]
    n_sub = min(n_sub, n_items)

    count_pct = np.zeros(n_heads, dtype=np.float64)
    count_k = np.zeros(n_heads, dtype=np.float64)

    for _ in range(n_bootstrap):
        idx = rng.choice(n_items, size=n_sub, replace=False)
        scores = _bridge_score_from_rows(fh_mat, th_mat, sh_mat, idx, score_fn=score_fn)
        for h in top_percent_set(scores, top_percent):
            count_pct[h] += 1
        if top_k > 0:
            for h in top_k_set(scores, top_k):
                count_k[h] += 1

    return {
        "stability_top_percent": count_pct / n_bootstrap,
        "stability_top_k": count_k / n_bootstrap,
    }


def subsampling_stability_mh(
    mh_mat: np.ndarray,
    n_sub: int,
    n_bootstrap: int,
    seed: int,
    top_percent: float,
    top_k: int = 0,
) -> dict[str, np.ndarray]:
    """Same as :func:`subsampling_stability` but for a single MH score matrix."""
    rng = np.random.default_rng(seed)
    n_items = mh_mat.shape[0]
    n_heads = mh_mat.shape[1]
    n_sub = min(n_sub, n_items)

    count_pct = np.zeros(n_heads, dtype=np.float64)
    count_k = np.zeros(n_heads, dtype=np.float64)

    for _ in range(n_bootstrap):
        idx = rng.choice(n_items, size=n_sub, replace=False)
        scores = mh_mat[idx].mean(axis=0)
        for h in top_percent_set(scores, top_percent):
            count_pct[h] += 1
        if top_k > 0:
            for h in top_k_set(scores, top_k):
                count_k[h] += 1

    return {
        "stability_top_percent": count_pct / n_bootstrap,
        "stability_top_k": count_k / n_bootstrap,
    }


# ---------------------------------------------------------------------------
# Overlap metrics
# ---------------------------------------------------------------------------

def jaccard(set_a: set[int], set_b: set[int]) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def enrichment(set_a: set[int], set_b: set[int], n_total: int) -> float:
    """Observed / expected intersection size under random baseline."""
    expected = len(set_a) * len(set_b) / n_total
    observed = len(set_a & set_b)
    if expected < 1e-12:
        return float("inf") if observed > 0 else 1.0
    return observed / expected


def hypergeometric_pvalue(set_a: set[int], set_b: set[int], n_total: int) -> float:
    """P(X >= observed) under the hypergeometric distribution.

    Models drawing |set_a| heads from n_total without replacement; set_b is
    the 'success population'.
    """
    k = len(set_a & set_b)      # observed successes
    M = n_total                  # population size
    n = len(set_b)               # successes in population
    N = len(set_a)               # draws
    # P(X >= k)
    return float(stats.hypergeom.sf(k - 1, M, n, N))


def spearman_corr(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Spearman rank correlation between two head-score vectors."""
    result = stats.spearmanr(scores_a, scores_b)
    return float(result.statistic)


def compute_overlap_metrics(
    name_a: str,
    set_a: set[int],
    score_a: np.ndarray,
    name_b: str,
    set_b: set[int],
    score_b: np.ndarray,
    n_total: int,
) -> dict[str, Any]:
    """Compute all six overlap metrics between two head sets / score vectors."""
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return {
        "set_a": name_a,
        "set_b": name_b,
        "size_a": len(set_a),
        "size_b": len(set_b),
        "intersection": inter,
        "union": union,
        "jaccard": jaccard(set_a, set_b),
        "enrichment": enrichment(set_a, set_b, n_total),
        "hypergeometric_pvalue": hypergeometric_pvalue(set_a, set_b, n_total),
        "spearman": spearman_corr(score_a, score_b),
    }
