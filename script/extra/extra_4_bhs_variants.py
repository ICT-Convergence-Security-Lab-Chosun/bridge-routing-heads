"""Extra 4: Alternative BHS Formula Variants — Head Set Discovery & Jaccard Overlap.

Re-uses the per-item gradient scores computed in Step 4
(output/step4_filtering_Bridge_head_Score/{model_short}/{lang}/scores_{cond}_{lang}.parquet)
and re-applies them with three modified aggregation formulas:

  Variant A — th-sh     : z(TH) - z(SH)          (FH removed)
  Variant B — fhth-2sh  : z(FH) + z(TH) - 2·z(SH) (SH double-weighted)
  Variant C — th        : z(TH)                   (naive TH-only baseline)

Pipeline per variant (mirrors Step 4 aggregation):
  1.  Load mean scores from Step-4 parquet files.
  2.  Compute per-language BHS vector with the variant formula.
  3.  Build top-percent candidate pool per language:
        LLaMA (llama31_70) : top 10 %
        Qwen  (qwen25_72)  : top 13 %
  4.  General head pool  = strict intersection across all specified languages.
  5.  Specific set       = top 4 % of each language AFTER masking General heads.
  6.  Within-variant Jaccard / enrichment / hypergeometric / Spearman overlap.

Intermediate per-head score DataFrames (raw scores, z-scores, BHS, rank) are
saved for every variant × language, so downstream scripts can load them directly.

Output layout
-------------
  output/extra/extra_4_bhs_variants/{model_short}/{variant}/
      {lang}/
          bridge_scores_{lang}.parquet   # per-head: S_FH, S_TH, S_SH, z_*, BHS, rank
          bridge_scores_{lang}.jsonl
      head_sets.json           # General + Specific + per-lang top-percent sets
      head_sets.jsonl
      overlap_metrics.json
      overlap_metrics.csv
      summary.csv
  output/extra/extra_4_bhs_variants/{model_short}/
      variant_comparison.json  # cross-variant Jaccard on General / Specific sets
      variant_comparison.csv

Stages
------
  variants  — compute all variant head sets (default)
  compare   — compare variant head sets vs. original Step-4 fh+th-sh head sets
              (Jaccard on set intersections, Spearman ρ on per-head BHS vectors)
  all       — run variants then compare

Output layout (compare stage)
-----------------------------
  output/extra/extra_4_bhs_variants/{model_short}/
      step4_comparison.json  — full records
      step4_comparison.csv   — tabular: variant, set_name, lang, jaccard,
                               intersection, size_variant, size_step4, spearman_bhs

Usage examples
--------------
# All variants, all models, all languages
python script/extra/extra_4_bhs_variants.py

# Single model
python script/extra/extra_4_bhs_variants.py --models llama31_70

# Custom top-percent overrides
python script/extra/extra_4_bhs_variants.py \\
    --top-percent-llama 0.08 --top-percent-qwen 0.13 --specific-percent 0.04

# Only specific variants
python script/extra/extra_4_bhs_variants.py --variants th-sh th

# Run variants stage then compare to Step-4 baseline
python script/extra/extra_4_bhs_variants.py --stage all

# Only the comparison (requires variant outputs to already exist)
python script/extra/extra_4_bhs_variants.py --stage compare

  ablation  — mean-ablation with k heads randomly drawn from each variant's pool.
              k = Step-5 final_bridge_heads count (General and per-lang Specific).
              30 random draws per condition; delta-NLL averaged across draws.
              Requires --model (HuggingFace path or local dir).
              Calibration: same as Step 6 — 50 correct records × (en/ko/zh/ja/es).
              No random baseline is produced.

Output layout (ablation stage)
------------------------------
  output/extra/extra_4_bhs_variants/{model_short}/ablation/
      {variant}_{lang}_general_draws.csv   — 30-row draw-level delta-NLL
      {variant}_{lang}_specific_draws.csv
      ablation_summary.csv                 — mean/std/95%CI over 30 draws

Usage example (ablation)
------------------------
python script/extra/extra_4_bhs_variants.py \\
    --stage ablation \\
    --models llama31_70 \\
    --model meta-llama/Llama-3.1-70B
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STEP4_ROOT  = PROJECT_ROOT / "output" / "step4_filtering_Bridge_head_Score"
STEP5_ROOT  = PROJECT_ROOT / "output" / "step5_BridgeHead_Ablation_Patchscopes"
STEP6_ROOT  = PROJECT_ROOT / "output" / "step6_Bridge_Head_validation"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "extra" / "extra_4_bhs_variants"

# ---------------------------------------------------------------------------
# Model-specific defaults
# ---------------------------------------------------------------------------
_TOP_PERCENT_DEFAULTS: dict[str, float] = {
    "llama31_70": 0.10,
    "qwen25_72":  0.13,
}
_DEFAULT_SPECIFIC_PERCENT = 0.04
_DEFAULT_LANGS = ["ko", "zh", "ja", "es"]  # "en" excluded by default (non-target)

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------
VARIANTS: dict[str, str] = {
    "th-sh":    "z(TH) - z(SH)",
    "fhth-2sh": "z(FH) + z(TH) - 2·z(SH)",
    "th":       "z(TH)",
}


# ---------------------------------------------------------------------------
# Math helpers  (replicate bridge_utils to keep this script self-contained)
# ---------------------------------------------------------------------------

def zscores(arr: np.ndarray) -> np.ndarray:
    mu, sigma = arr.mean(), arr.std()
    if sigma < 1e-12:
        return np.zeros_like(arr)
    return (arr - mu) / sigma


def compute_variant(
    mean_fh: np.ndarray,
    mean_th: np.ndarray,
    mean_sh: np.ndarray,
    variant: str,
) -> np.ndarray:
    """Return the BHS vector for the requested variant formula."""
    z_fh = zscores(mean_fh)
    z_th = zscores(mean_th)
    z_sh = zscores(mean_sh)
    if variant == "th-sh":
        return z_th - z_sh
    if variant == "fhth-2sh":
        return z_fh + z_th - 2.0 * z_sh
    if variant == "th":
        return z_th
    raise ValueError(f"Unknown variant: {variant!r}")


def top_percent_set(scores: np.ndarray, percent: float) -> set[int]:
    k = max(1, int(len(scores) * percent))
    return set(int(i) for i in np.argsort(scores)[::-1][:k])


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _jsonl_write(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_mean_scores(
    model_short: str,
    lang: str,
    condition: str,
) -> np.ndarray | None:
    """Load a condition parquet and return the per-head mean vector."""
    path = STEP4_ROOT / model_short / lang / f"scores_{condition}_{lang}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    matrix = df.drop(columns=["item_id"]).values.astype(np.float32)
    return matrix.mean(axis=0)


def infer_arch(model_short: str, langs: list[str]) -> tuple[int, int]:
    """Infer (n_layers, n_heads) from the first available parquet."""
    for lang in langs:
        for cond in ("FH", "TH", "SH"):
            path = STEP4_ROOT / model_short / lang / f"scores_{cond}_{lang}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                cols = [c for c in df.columns if c != "item_id"]
                n_layers = max(int(c.split("H")[0][1:]) for c in cols) + 1
                n_heads  = max(int(c.split("H")[1])     for c in cols) + 1
                return n_layers, n_heads
    raise FileNotFoundError(f"No Step-4 parquet found for {model_short}")


# ---------------------------------------------------------------------------
# Overlap metrics  (self-contained)
# ---------------------------------------------------------------------------

from scipy import stats as _scipy_stats


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def enrichment(a: set[int], b: set[int], n_total: int) -> float:
    expected = len(a) * len(b) / n_total
    if expected < 1e-12:
        return float("inf") if a & b else 1.0
    return len(a & b) / expected


def hypergeometric_pvalue(a: set[int], b: set[int], n_total: int) -> float:
    k = len(a & b)
    return float(_scipy_stats.hypergeom.sf(k - 1, n_total, len(b), len(a)))


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return float(_scipy_stats.spearmanr(x, y).statistic)


def overlap_record(
    name_a: str, set_a: set[int], score_a: np.ndarray,
    name_b: str, set_b: set[int], score_b: np.ndarray,
    n_total: int,
) -> dict:
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


# ---------------------------------------------------------------------------
# Per-head DataFrame builder
# ---------------------------------------------------------------------------

def build_score_df(
    mean_fh: np.ndarray,
    mean_th: np.ndarray,
    mean_sh: np.ndarray,
    bhs: np.ndarray,
    n_layers: int,
    n_heads: int,
    model_short: str,
    lang: str,
    variant: str,
) -> pd.DataFrame:
    z_fh = zscores(mean_fh)
    z_th = zscores(mean_th)
    z_sh = zscores(mean_sh)
    ranks = (-bhs).argsort().argsort() + 1  # rank 1 = highest BHS

    rows = []
    for flat_idx in range(n_layers * n_heads):
        layer, head = divmod(flat_idx, n_heads)
        rows.append({
            "model":   model_short,
            "lang":    lang,
            "variant": variant,
            "layer":   layer,
            "head":    head,
            "head_id": f"L{layer}H{head}",
            "flat_index": flat_idx,
            "S_FH":    float(mean_fh[flat_idx]),
            "S_TH":    float(mean_th[flat_idx]),
            "S_SH":    float(mean_sh[flat_idx]),
            "z_FH":    float(z_fh[flat_idx]),
            "z_TH":    float(z_th[flat_idx]),
            "z_SH":    float(z_sh[flat_idx]),
            "BHS":     float(bhs[flat_idx]),
            "rank_bhs": int(ranks[flat_idx]),
        })
    return pd.DataFrame(rows)


def flat_set_to_records(
    flat_indices: set[int],
    bhs: np.ndarray,
    n_heads: int,
) -> list[dict]:
    result = []
    for flat_idx in sorted(flat_indices, key=lambda f: -bhs[f]):
        layer, head = divmod(flat_idx, n_heads)
        result.append({
            "layer": layer,
            "head":  head,
            "head_id": f"L{layer}H{head}",
            "flat_index": flat_idx,
            "bhs_score": float(bhs[flat_idx]),
            "rank": int((-bhs).argsort().argsort()[flat_idx] + 1),
        })
    return result


# ---------------------------------------------------------------------------
# Core: run one variant for one model
# ---------------------------------------------------------------------------

def run_variant(
    model_short: str,
    variant: str,
    langs: list[str],
    top_percent: float,
    specific_percent: float,
    out_model_dir: Path,
) -> dict[str, set[int]]:
    """Run one variant; return flat_sets dict for cross-variant comparison.

    Returns {set_name: set_of_flat_indices}.
    """
    variant_dir = out_model_dir / variant
    formula_label = VARIANTS[variant]
    print(f"  variant={variant!r}  formula={formula_label!r}  top%={top_percent:.0%}  specific%={specific_percent:.0%}")

    n_layers, n_heads = infer_arch(model_short, langs)
    n_total = n_layers * n_heads

    # --- Load mean scores per language ---
    raw: dict[str, dict[str, np.ndarray]] = {}
    for lang in langs:
        mean_fh = load_mean_scores(model_short, lang, "FH")
        mean_th = load_mean_scores(model_short, lang, "TH")
        mean_sh = load_mean_scores(model_short, lang, "SH")
        if any(v is None for v in (mean_fh, mean_th, mean_sh)):
            missing = [c for c, v in zip(["FH", "TH", "SH"], [mean_fh, mean_th, mean_sh]) if v is None]
            print(f"    [WARN] {model_short}/{lang}: missing conditions {missing} — skipping.")
            continue
        raw[lang] = {"fh": mean_fh, "th": mean_th, "sh": mean_sh}

    langs_present = [l for l in langs if l in raw]
    if not langs_present:
        print(f"    [WARN] No data found for {model_short} — skipping variant.")
        return {}

    # --- Compute BHS vectors and per-lang score DataFrames ---
    bhs_by_lang: dict[str, np.ndarray] = {}
    for lang in langs_present:
        s = raw[lang]
        bhs = compute_variant(s["fh"], s["th"], s["sh"], variant)
        bhs_by_lang[lang] = bhs

        df = build_score_df(
            s["fh"], s["th"], s["sh"], bhs,
            n_layers, n_heads, model_short, lang, variant,
        )
        lang_dir = variant_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        pq_path = lang_dir / f"bridge_scores_{lang}.parquet"
        df.to_parquet(pq_path, index=False)
        _jsonl_write(df.to_dict(orient="records"), pq_path.with_suffix(".jsonl"))
        print(f"    [{lang}] Saved bridge_scores  (BHS range [{bhs.min():.3f}, {bhs.max():.3f}])")

    avg_bhs = np.mean([bhs_by_lang[l] for l in langs_present], axis=0)

    # --- Build head sets ---
    flat_sets: dict[str, set[int]] = {}
    head_sets_out: dict[str, list[dict]] = {}

    def _register(name: str, indices: set[int], score_vec: np.ndarray) -> None:
        flat_sets[name] = indices
        head_sets_out[name] = flat_set_to_records(indices, score_vec, n_heads)

    # Per-language top-percent pool
    bridge_top: dict[str, set[int]] = {}
    for lang in langs_present:
        top_set = top_percent_set(bhs_by_lang[lang], top_percent)
        bridge_top[lang] = top_set
        _register(f"H_Bridge_{lang.upper()}", top_set, bhs_by_lang[lang])

    # General = strict intersection
    if len(bridge_top) >= 2:
        general: set[int] = set.intersection(*bridge_top.values())
    elif len(bridge_top) == 1:
        general = next(iter(bridge_top.values())).copy()
    else:
        general = set()
    _register("H_Bridge_General", general, avg_bhs)
    print(f"    H_Bridge_General: {len(general)} heads  (intersection of {len(langs_present)} langs)")

    # Specific = top specific_percent per language, after masking General
    for lang in langs_present:
        bhs = bhs_by_lang[lang]
        masked = bhs.copy()
        for h in general:
            masked[h] = -np.inf
        specific = top_percent_set(masked, specific_percent)
        _register(f"H_Bridge_Specific_{lang.upper()}", specific, bhs)
        print(f"    H_Bridge_Specific_{lang.upper()}: {len(specific)} heads")

    # --- Save head sets ---
    variant_dir.mkdir(parents=True, exist_ok=True)
    _save_json(head_sets_out, variant_dir / "head_sets.json")
    _jsonl_write(
        [{"set_name": k, "heads": v} for k, v in head_sets_out.items()],
        variant_dir / "head_sets.jsonl",
    )

    # --- Overlap metrics ---
    def _score_for(sname: str) -> np.ndarray:
        if sname == "H_Bridge_General":
            return avg_bhs
        if sname.startswith("H_Bridge_Specific_"):
            lc = sname.removeprefix("H_Bridge_Specific_").lower()
            return bhs_by_lang.get(lc, np.zeros(n_total))
        lc = sname.removeprefix("H_Bridge_").lower()
        return bhs_by_lang.get(lc, np.zeros(n_total))

    pairs: list[tuple[str, str]] = []
    for lang in langs_present:
        pairs.append(("H_Bridge_General", f"H_Bridge_Specific_{lang.upper()}"))
    for i, la in enumerate(langs_present):
        for lb in langs_present[i + 1:]:
            pairs.append((f"H_Bridge_{la.upper()}", f"H_Bridge_{lb.upper()}"))

    ov_records = [
        overlap_record(
            na, flat_sets[na], _score_for(na),
            nb, flat_sets[nb], _score_for(nb),
            n_total,
        )
        for na, nb in pairs
        if flat_sets.get(na) and flat_sets.get(nb)
    ]
    _save_json(ov_records, variant_dir / "overlap_metrics.json")
    if ov_records:
        df_ov = pd.DataFrame(ov_records)
        df_ov.to_csv(variant_dir / "overlap_metrics.csv", index=False)
        summary_cols = ["set_a", "set_b", "intersection", "jaccard", "enrichment",
                        "hypergeometric_pvalue", "spearman"]
        df_ov[summary_cols].sort_values("jaccard", ascending=False).to_csv(
            variant_dir / "summary.csv", index=False,
        )
        print(f"    Overlap matrix ({len(ov_records)} pairs) saved.")

    return flat_sets


# ---------------------------------------------------------------------------
# Cross-variant comparison
# ---------------------------------------------------------------------------

def run_cross_variant_comparison(
    model_short: str,
    langs_present: list[str],
    per_variant_flat_sets: dict[str, dict[str, set[int]]],
    out_model_dir: Path,
) -> None:
    """Compare General and Specific head sets across all variant pairs."""
    variant_names = list(per_variant_flat_sets.keys())
    if len(variant_names) < 2:
        return

    n_layers, n_heads = infer_arch(model_short, langs_present)
    n_total = n_layers * n_heads

    compare_set_names = ["H_Bridge_General"]
    for lang in langs_present:
        compare_set_names.append(f"H_Bridge_Specific_{lang.upper()}")
    for lang in langs_present:
        compare_set_names.append(f"H_Bridge_{lang.upper()}")

    records: list[dict] = []
    # Dummy uniform score vector for cross-variant comparisons
    dummy = np.ones(n_total, dtype=np.float32)

    for i, va in enumerate(variant_names):
        for vb in variant_names[i + 1:]:
            sets_a = per_variant_flat_sets[va]
            sets_b = per_variant_flat_sets[vb]
            for sname in compare_set_names:
                sa = sets_a.get(sname, set())
                sb = sets_b.get(sname, set())
                if not sa or not sb:
                    continue
                rec = overlap_record(
                    f"{sname} [{va}]", sa, dummy,
                    f"{sname} [{vb}]", sb, dummy,
                    n_total,
                )
                rec["variant_a"] = va
                rec["variant_b"] = vb
                rec["set_name"] = sname
                records.append(rec)

    if not records:
        return

    _save_json(records, out_model_dir / "variant_comparison.json")
    df = pd.DataFrame(records)
    df.to_csv(out_model_dir / "variant_comparison.csv", index=False)
    print(f"\n  Cross-variant comparison: {len(records)} pairs saved → variant_comparison.*")

    # Print readable summary
    summary = df[["variant_a", "variant_b", "set_name", "intersection", "jaccard"]].sort_values(
        ["variant_a", "variant_b", "set_name"]
    )
    print(summary.to_string(index=False))


# ---------------------------------------------------------------------------
# Step-4 baseline comparison
# ---------------------------------------------------------------------------

_STEP4_FORMULA = "fh+th-sh"   # the original Step-4 formula directory name


def _load_step4_head_sets(model_short: str) -> dict[str, set[int]] | None:
    """Load Step-4 fh+th-sh head_sets.json → {set_name: set of flat indices}."""
    path = STEP4_ROOT / model_short / _STEP4_FORMULA / "head_sets.json"
    if not path.exists():
        print(f"  [WARN] Step-4 head_sets not found: {path}")
        return None
    raw = json.loads(path.read_text())
    result: dict[str, set[int]] = {}
    for sname, recs in raw.items():
        result[sname] = {int(r["flat_head_index"]) for r in recs}
    return result


def _load_step4_bhs(model_short: str, lang: str) -> np.ndarray | None:
    """Load Step-4 fh+th-sh bridge_scores parquet → BHS vector (n_total,)."""
    path = STEP4_ROOT / model_short / _STEP4_FORMULA / lang / f"bridge_scores_{lang}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df["BHS"].values.astype(np.float64)


def _load_variant_head_sets(out_model_dir: Path, variant: str) -> dict[str, set[int]] | None:
    """Load variant head_sets.json → {set_name: set of flat indices}."""
    path = out_model_dir / variant / "head_sets.json"
    if not path.exists():
        print(f"  [WARN] Variant head_sets not found: {path}")
        return None
    raw = json.loads(path.read_text())
    result: dict[str, set[int]] = {}
    for sname, recs in raw.items():
        result[sname] = {int(r["flat_index"]) for r in recs}
    return result


def _load_variant_bhs(out_model_dir: Path, variant: str, lang: str) -> np.ndarray | None:
    """Load variant bridge_scores parquet → BHS vector (n_total,)."""
    path = out_model_dir / variant / lang / f"bridge_scores_{lang}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df["BHS"].values.astype(np.float64)


def run_step4_comparison(
    model_short: str,
    variants: list[str],
    langs: list[str],
    out_model_dir: Path,
) -> None:
    """Compare each variant's head sets and BHS vectors to the Step-4 fh+th-sh baseline.

    For every (variant, set_name) pair that exists in both the variant outputs
    and the Step-4 results, computes:
      - Jaccard similarity between the two head sets
      - intersection / union sizes
      - Spearman ρ on the per-head BHS vectors (per language only)

    Results saved to {out_model_dir}/step4_comparison.json and .csv.
    """
    print(f"\n{'='*60}")
    print(f"Step-4 comparison — {model_short}")
    print(f"{'='*60}")

    n_layers, n_heads = infer_arch(model_short, langs)
    n_total = n_layers * n_heads

    # Load Step-4 baseline
    s4_sets = _load_step4_head_sets(model_short)
    if s4_sets is None:
        print("  Aborting comparison: Step-4 data missing.")
        return

    records: list[dict] = []

    for variant in variants:
        v_sets = _load_variant_head_sets(out_model_dir, variant)
        if v_sets is None:
            print(f"  [{variant}] Skipping — variant outputs not found. Run --stage variants first.")
            continue

        # ---- Set-level Jaccard (all named sets) ----
        for sname in sorted(v_sets):
            s4_set = s4_sets.get(sname)
            v_set  = v_sets[sname]
            if s4_set is None:
                # Step-4 may not have the same lang-coverage; skip
                continue

            inter = len(v_set & s4_set)
            union = len(v_set | s4_set)
            jac   = inter / union if union else 1.0

            rec: dict[str, Any] = {
                "model":       model_short,
                "variant":     variant,
                "step4_formula": _STEP4_FORMULA,
                "set_name":    sname,
                "size_variant": len(v_set),
                "size_step4":   len(s4_set),
                "intersection": inter,
                "union":        union,
                "jaccard":      round(jac, 6),
                "spearman_bhs": None,   # filled below for per-lang sets
            }

            # ---- Per-language Spearman on BHS vectors ----
            # Determine which language this set belongs to (if any)
            lang_of_set: str | None = None
            for l in langs:
                if sname in (f"H_Bridge_{l.upper()}",
                             f"H_Bridge_Specific_{l.upper()}"):
                    lang_of_set = l
                    break

            if lang_of_set is not None:
                v_bhs  = _load_variant_bhs(out_model_dir, variant, lang_of_set)
                s4_bhs = _load_step4_bhs(model_short, lang_of_set)
                if v_bhs is not None and s4_bhs is not None:
                    rho = spearman_corr(v_bhs, s4_bhs)
                    rec["spearman_bhs"] = round(rho, 6)

            records.append(rec)

        # ---- General set: compute Spearman using average BHS over all langs ----
        # (Find the General row already appended and fill spearman from avg vectors)
        v_avg_bhs_arrays = [
            _load_variant_bhs(out_model_dir, variant, l)
            for l in langs
            if _load_variant_bhs(out_model_dir, variant, l) is not None
        ]
        s4_avg_bhs_arrays = [
            _load_step4_bhs(model_short, l)
            for l in langs
            if _load_step4_bhs(model_short, l) is not None
        ]
        if v_avg_bhs_arrays and s4_avg_bhs_arrays:
            v_avg  = np.mean(v_avg_bhs_arrays,  axis=0)
            s4_avg = np.mean(s4_avg_bhs_arrays, axis=0)
            rho_gen = spearman_corr(v_avg, s4_avg)
            for rec in records:
                if rec["variant"] == variant and rec["set_name"] == "H_Bridge_General":
                    rec["spearman_bhs"] = round(rho_gen, 6)

    if not records:
        print("  No comparison records produced.")
        return

    # Save
    _save_json(records, out_model_dir / "step4_comparison.json")
    df = pd.DataFrame(records)
    col_order = [
        "model", "variant", "step4_formula", "set_name",
        "size_variant", "size_step4", "intersection", "union",
        "jaccard", "spearman_bhs",
    ]
    df = df[[c for c in col_order if c in df.columns]]
    df.to_csv(out_model_dir / "step4_comparison.csv", index=False)

    # Print readable summary
    print()
    summary = df[["variant", "set_name", "size_variant", "size_step4",
                  "intersection", "jaccard", "spearman_bhs"]].sort_values(
        ["variant", "set_name"]
    )
    print(summary.to_string(index=False))
    print(f"\n  Saved → {out_model_dir}/step4_comparison.[json|csv]")


# ---------------------------------------------------------------------------
# Variant Mean Ablation  (stage: ablation)
# ---------------------------------------------------------------------------

def _load_step5_k_values(
    model_short: str,
    langs: list[str],
) -> tuple[int, dict[str, int]]:
    """Return (k_general, {lang: k_specific}) from Step-5 final_bridge_heads.json."""
    path = STEP5_ROOT / model_short / "final_bridge_heads.json"
    if not path.exists():
        print(f"  [WARN] Step-5 final_bridge_heads not found: {path}")
        return 0, {}
    raw = json.loads(path.read_text())["final_bridge_heads"]
    k_general = len(raw.get("general", []))
    print(f"  k_general : {k_general}")
    k_specific = {
        lang: len(raw.get("specific", {}).get(lang, []))
        for lang in langs
    }
    for lang, k in k_specific.items():
        print(f"  k_specific[{lang}] : {k}")
    return k_general, k_specific


def _variant_flat_pools(
    out_model_dir: Path,
    variant: str,
    langs: list[str],
) -> dict[str, list[int]]:
    """Load variant head_sets.json → {'general': [...], 'specific_ko': [...], ...}.

    Keys: 'general' and 'specific_{lang}' for each lang.
    Values: sorted flat head indices.
    """
    hs_path = out_model_dir / variant / "head_sets.json"
    if not hs_path.exists():
        return {}
    raw = json.loads(hs_path.read_text())
    pools: dict[str, list[int]] = {}
    if "H_Bridge_General" in raw:
        pools["general"] = sorted(int(r["flat_index"]) for r in raw["H_Bridge_General"])
    for lang in langs:
        key = f"H_Bridge_Specific_{lang.upper()}"
        if key in raw:
            pools[f"specific_{lang}"] = sorted(
                int(r["flat_index"]) for r in raw[key]
            )
    return pools


def _load_step6_bth_delta(model_short: str, lang: str, set_type: str) -> float | None:
    """Return Step-6 bridge-head ablation delta_nll_mean for comparison, or None."""
    csv = STEP6_ROOT / model_short / "ablation_all_langs.csv"
    if not csv.exists():
        return None
    try:
        df = pd.read_csv(csv)
        row = df[
            (df["lang"] == lang)
            & (df["head_set_type"] == set_type)
            & (df["baseline_type"] == "bth")
        ]
        if len(row):
            return float(row.iloc[0]["delta_nll_mean"])
    except Exception:
        pass
    return None


def run_variant_ablation(
    model_short: str,
    variants: list[str],
    langs: list[str],
    out_model_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Mean-ablation with k heads randomly drawn from each variant's pool.

    k = Step-5 final_bridge_heads count (general / specific per lang).
    Each condition is repeated n_random_draws times and the delta-NLL is averaged.
    Calibration: identical to Step 6 — 50 correct records x (en, ko, zh, ja, es).
    """
    sys.path.insert(0, str(PROJECT_ROOT / "script"))
    from utils.model_utils import load_model_and_tokenizer
    from utils.prompt_utils import wrap_prompt
    from utils.bridge_utils import extract_answer_span, load_filtered_records, sample_records
    from utils.head_hooks import (
        HeadMaskManager,
        _get_head_dim, _get_num_heads, _get_num_layers,
        collect_mean_head_outputs, compute_nll_eval,
    )

    print(f"\n{'='*60}")
    print(f"Variant Mean Ablation — {model_short}")
    print(f"{'='*60}")
    print(f"  variants       : {variants}")
    print(f"  langs          : {langs}")
    print(f"  ablation_n     : {args.ablation_sample_size} records/lang")
    print(f"  calibration_n  : {args.calibration_size} x 5 langs")
    print(f"  n_random_draws : {args.n_random_draws}")

    if not getattr(args, "model", None):
        print("  [ERROR] --model is required for --stage ablation. Skipping.")
        return

    # ----------------------------------------------------------------
    # Load model
    # ----------------------------------------------------------------
    print(f"  Loading model  : {args.model}")
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        device="auto",
        torch_dtype=getattr(args, "torch_dtype", "auto"),
        hf_token=getattr(args, "hf_token", None),
        trust_remote_code=getattr(args, "trust_remote_code", False),
        attn_implementation="eager",
    )
    n_layers = _get_num_layers(model)
    n_heads  = _get_num_heads(model)
    head_dim = _get_head_dim(model)
    print(f"  n_layers={n_layers}  n_heads={n_heads}  head_dim={head_dim}")

    # ----------------------------------------------------------------
    # Calibration means  (same protocol as Step 6)
    # ----------------------------------------------------------------
    CALIB_LANGS = ["en", "ko", "ja", "zh", "es"]
    input_root = PROJECT_ROOT / "data"
    calib_prompts: list[str] = []
    for clang in CALIB_LANGS:
        try:
            recs = load_filtered_records(input_root, model_short, clang)
        except FileNotFoundError:
            print(f"  [WARN] Calibration: no correct records for lang={clang} — skipping.")
            continue
        for rec in sample_records(recs, args.calibration_size, args.seed):
            calib_prompts.append(wrap_prompt(rec["prompts"]["two_hop"], rec["lang"]))
    print(f"  Calibration: {len(calib_prompts)} prompts collected.")
    mean_values = collect_mean_head_outputs(
        model, tokenizer, calib_prompts, n_layers, n_heads, head_dim
    )
    print("  Calibration means ready.")

    # ----------------------------------------------------------------
    # Step-5 k values
    # ----------------------------------------------------------------
    k_general, k_specific = _load_step5_k_values(model_short, langs)
    print(f"  Step-5 k_general={k_general}  k_specific={k_specific}")

    ablation_dir = out_model_dir / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    all_summary: list[dict] = []

    # ----------------------------------------------------------------
    # Per-variant ablation
    # ----------------------------------------------------------------
    for variant in variants:
        print(f"\n  --- variant={variant} ({VARIANTS[variant]}) ---")

        pools = _variant_flat_pools(out_model_dir, variant, langs)
        if not pools:
            print(f"  [WARN] No head_sets.json found for variant={variant}. "
                  "Run --stage variants first.")
            continue

        for lang in langs:
            # ---- Load correct records + compute base NLLs ----------
            try:
                records = load_filtered_records(input_root, model_short, lang)
            except FileNotFoundError:
                print(f"    [{lang}] No correct records — skipping.")
                continue
            records = sample_records(records, args.ablation_sample_size, args.seed)

            items: list[dict] = []
            for rec in records:
                gold = extract_answer_span(
                    rec.get("eval", {}).get("two_hop_pred", ""), mode="first_span"
                )
                if not gold:
                    continue
                prompt = wrap_prompt(rec["prompts"]["two_hop"], rec["lang"])
                nll_base = compute_nll_eval(model, tokenizer, prompt, gold)
                if math.isnan(nll_base):
                    continue
                items.append({"id": rec["id"], "prompt": prompt,
                               "gold": gold, "nll_base": nll_base})

            if not items:
                print(f"    [{lang}] 0 valid items — skipping.")
                continue
            print(f"    [{lang}] {len(items)}/{len(records)} valid items")

            # ---- Per pool-type (general / specific) -----------------
            for set_type, pool_key, k in [
                ("general",  "general",          k_general),
                ("specific", f"specific_{lang}",  k_specific.get(lang, 0)),
            ]:
                pool_list = pools.get(pool_key, [])
                if not pool_list:
                    print(f"      [{lang}/{set_type}] pool is empty — skipping.")
                    continue
                if k == 0:
                    print(f"      [{lang}/{set_type}] k=0 (no Step-5 heads) — skipping.")
                    continue

                with_replacement = k > len(pool_list)
                if with_replacement:
                    print(f"      [{lang}/{set_type}] WARNING: pool={len(pool_list)} < k={k}; "
                          "sampling with replacement.")

                # ---- n_random_draws draws ---------------------------
                draw_rows: list[dict] = []
                draw_delta_means: list[float] = []

                with HeadMaskManager(model, n_layers, n_heads, head_dim) as mask_mgr:
                    for draw_i in range(args.n_random_draws):
                        if with_replacement:
                            sampled_flat = rng.choices(pool_list, k=k)
                        else:
                            sampled_flat = rng.sample(pool_list, k)
                        head_set = [divmod(f, n_heads) for f in sampled_flat]

                        item_deltas: list[float] = []
                        for item in items:
                            mask_mgr.reset_masks()
                            mask_mgr.apply_head_set_ablation(head_set, mean_values)
                            nll_a = compute_nll_eval(
                                model, tokenizer, item["prompt"], item["gold"]
                            )
                            item_deltas.append(nll_a - item["nll_base"])
                        mask_mgr.reset_masks()

                        draw_mean = float(sum(item_deltas) / len(item_deltas))
                        draw_delta_means.append(draw_mean)
                        draw_rows.append({
                            "model":          model_short,
                            "variant":        variant,
                            "lang":           lang,
                            "set_type":       set_type,
                            "draw":           draw_i,
                            "k":              k,
                            "pool_size":      len(pool_list),
                            "n_items":        len(item_deltas),
                            "delta_nll_mean": round(draw_mean, 6),
                        })
                        if (draw_i + 1) % 10 == 0 or draw_i == args.n_random_draws - 1:
                            running = sum(draw_delta_means) / len(draw_delta_means)
                            print(f"      [{lang}/{set_type}] draw {draw_i+1}/{args.n_random_draws}"
                                  f"  running_mean={running:.4f}")

                # Save draw-level CSV
                pd.DataFrame(draw_rows).to_csv(
                    ablation_dir / f"{variant}_{lang}_{set_type}_draws.csv", index=False
                )

                # Summary stats
                arr = np.array(draw_delta_means, dtype=np.float64)
                ci_lo = float(np.percentile(arr, 2.5))
                ci_hi = float(np.percentile(arr, 97.5))
                step6_delta = _load_step6_bth_delta(model_short, lang, set_type)

                row = {
                    "model":                     model_short,
                    "variant":                   variant,
                    "lang":                      lang,
                    "set_type":                  set_type,
                    "k":                         k,
                    "pool_size":                 len(pool_list),
                    "n_random_draws":            args.n_random_draws,
                    "n_items_per_draw":          len(items),
                    "delta_nll_mean":            round(float(arr.mean()), 6),
                    "delta_nll_std":             round(float(arr.std()), 6),
                    "delta_nll_ci_lo_95":        round(ci_lo, 6),
                    "delta_nll_ci_hi_95":        round(ci_hi, 6),
                    "sampling_with_replacement": with_replacement,
                    "step6_bth_delta_nll":       step6_delta,
                }
                all_summary.append(row)
                s6_str = (f"  step6_bth={step6_delta:.4f}" if step6_delta is not None else "")
                print(f"      [{lang}/{set_type}] dNLL={arr.mean():.4f}+-{arr.std():.4f}"
                      f"  95%CI[{ci_lo:.4f},{ci_hi:.4f}]  k={k}  pool={len(pool_list)}{s6_str}")

    # ----------------------------------------------------------------
    # Save combined summary
    # ----------------------------------------------------------------
    if all_summary:
        df_s = pd.DataFrame(all_summary)
        df_s.to_csv(ablation_dir / "ablation_summary.csv", index=False)
        print(f"\n  Saved → {ablation_dir}/ablation_summary.csv")
        print()
        disp_cols = ["variant", "lang", "set_type", "k", "pool_size",
                     "delta_nll_mean", "delta_nll_std",
                     "delta_nll_ci_lo_95", "delta_nll_ci_hi_95",
                     "step6_bth_delta_nll"]
        print(df_s[[c for c in disp_cols if c in df_s.columns]].to_string(index=False))
    else:
        print("  No ablation results produced.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extra 4: alternative BHS formula variants and head-set Jaccard analysis."
    )
    p.add_argument(
        "--models", nargs="+",
        default=["llama31_70", "qwen25_72"],
        metavar="MODEL",
        help="Model short-names to process (default: llama31_70 qwen25_72).",
    )
    p.add_argument(
        "--langs", nargs="+",
        default=_DEFAULT_LANGS,
        metavar="LANG",
        help=f"Language codes (default: {_DEFAULT_LANGS}).",
    )
    p.add_argument(
        "--variants", nargs="+",
        default=list(VARIANTS),
        choices=list(VARIANTS),
        metavar="VARIANT",
        help="Which variants to run. Choices: th-sh  fhth-2sh  th  (default: all).",
    )
    p.add_argument(
        "--top-percent-llama", type=float, default=0.10,
        metavar="P",
        help="Top-percent pool size for llama31_70 (default: 0.10 = 10%%).",
    )
    p.add_argument(
        "--top-percent-qwen", type=float, default=0.13,
        metavar="P",
        help="Top-percent pool size for qwen25_72 (default: 0.13 = 13%%).",
    )
    p.add_argument(
        "--specific-percent", type=float, default=_DEFAULT_SPECIFIC_PERCENT,
        metavar="P",
        help=f"Specific head pool size after masking General (default: {_DEFAULT_SPECIFIC_PERCENT}).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        metavar="DIR",
        help="Override output root directory.",
    )
    p.add_argument(
        "--stage",
        choices=["variants", "compare", "ablation", "all"],
        default="variants",
        help=(
            "Which stage(s) to run: "
            "'variants'  = compute new formula head sets (default); "
            "'compare'   = Jaccard + Spearman rho vs. Step-4 fh+th-sh baseline "
            "(requires variant outputs to exist); "
            "'ablation'  = mean-ablation with random draws from variant pools "
            "(requires --model); "
            "'all'       = variants then compare then ablation."
        ),
    )
    # ----------------------------------------------------------------
    # Ablation-specific arguments
    # ----------------------------------------------------------------
    p.add_argument(
        "--model", default=None, metavar="HF_PATH",
        help="HuggingFace model ID or local path. Required for --stage ablation.",
    )
    p.add_argument(
        "--ablation-sample-size", type=int, default=100, metavar="N",
        help="Correct records per language for ablation NLL (default: 100).",
    )
    p.add_argument(
        "--calibration-size", type=int, default=50, metavar="N",
        help="Correct records per calibration language for mean collection (default: 50).",
    )
    p.add_argument(
        "--n-random-draws", type=int, default=30, metavar="N",
        help="Number of random pool subsamples to average per condition (default: 30).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hf-token", default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--torch-dtype", default="auto")
    return p.parse_args()


def _top_percent_for(model_short: str, args: argparse.Namespace) -> float:
    if model_short == "llama31_70":
        return args.top_percent_llama
    if model_short == "qwen25_72":
        return args.top_percent_qwen
    # Fall back to a reasonable default
    return _TOP_PERCENT_DEFAULTS.get(model_short, 0.10)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_root: Path = args.output_dir if args.output_dir is not None else OUTPUT_ROOT

    print("=" * 70)
    print("Extra 4 — BHS Formula Variants")
    print("=" * 70)
    print(f"  models    : {args.models}")
    print(f"  langs     : {args.langs}")
    print(f"  variants  : {args.variants}")
    print(f"  stage     : {args.stage}")
    print(f"  top%      : llama={args.top_percent_llama:.0%}  qwen={args.top_percent_qwen:.0%}")
    print(f"  specific% : {args.specific_percent:.0%}")
    if args.stage in ("ablation", "all"):
        print(f"  model     : {args.model}")
        print(f"  abl_n     : {args.ablation_sample_size}  "
              f"calib_n={args.calibration_size}  draws={args.n_random_draws}")
    print(f"  output    : {out_root}")
    print()

    for model_short in args.models:
        print(f"\n{'='*60}")
        print(f"Model: {model_short}")
        print(f"{'='*60}")
        out_model_dir = out_root / model_short
        top_percent = _top_percent_for(model_short, args)

        # ----------------------------------------------------------------
        # Stage: variants
        # ----------------------------------------------------------------
        if args.stage in ("variants", "all"):
            per_variant_flat_sets: dict[str, dict[str, set[int]]] = {}
            langs_seen: list[str] = []

            for variant in args.variants:
                print(f"\n--- {variant} ---")
                flat_sets = run_variant(
                    model_short=model_short,
                    variant=variant,
                    langs=args.langs,
                    top_percent=top_percent,
                    specific_percent=args.specific_percent,
                    out_model_dir=out_model_dir,
                )
                per_variant_flat_sets[variant] = flat_sets

                # Collect langs_seen from first successful variant
                if not langs_seen and flat_sets:
                    langs_seen = [
                        sname.removeprefix("H_Bridge_").lower()
                        for sname in flat_sets
                        if sname.startswith("H_Bridge_") and sname != "H_Bridge_General"
                        and not sname.startswith("H_Bridge_Specific_")
                    ]

            if len(per_variant_flat_sets) >= 2 and langs_seen:
                print(f"\n--- Cross-variant comparison ---")
                run_cross_variant_comparison(
                    model_short=model_short,
                    langs_present=langs_seen,
                    per_variant_flat_sets=per_variant_flat_sets,
                    out_model_dir=out_model_dir,
                )

        # ----------------------------------------------------------------
        # Stage: compare  (vs. Step-4 fh+th-sh baseline)
        # ----------------------------------------------------------------
        if args.stage in ("compare", "all"):
            run_step4_comparison(
                model_short=model_short,
                variants=args.variants,
                langs=args.langs,
                out_model_dir=out_model_dir,
            )

        # ----------------------------------------------------------------
        # Stage: ablation  (variant pool random sampling + mean ablation)
        # ----------------------------------------------------------------
        if args.stage in ("ablation", "all"):
            run_variant_ablation(
                model_short=model_short,
                variants=args.variants,
                langs=args.langs,
                out_model_dir=out_model_dir,
                args=args,
            )

    print("\n" + "=" * 70)
    print(f"Done. Results saved to: {out_root}")
    print("=" * 70)


if __name__ == "__main__":
    main()
