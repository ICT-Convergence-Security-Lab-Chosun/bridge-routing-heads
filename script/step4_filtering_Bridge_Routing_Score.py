"""Step 4: Bridge Routing Score (BRS) computation and Language-General/Specific head set discovery.

Computes per-item gradient-based head importance scores for three conditions:
  FH  — first-hop prompt  (bridge entity prediction)
  TH  — two-hop prompt    (final answer prediction)
  SH  — second-hop prompt (bridge entity given explicitly)

Then aggregates into two Bridge Routing Score (BRS) formula variants:
  fh_th_sh : z(FH) + z(TH) - z(SH)
  th_fh_sh : z(TH) - z(FH) - z(SH)

Head sets produced per formula:
  H_Bridge_{LANG}          — top-percent BRS heads for each language
  H_Bridge_General         — strict intersection of H_Bridge_{LANG} across all languages
  H_Bridge_Specific_{LANG} — top-percent BRS heads after masking out H_Bridge_General

Overlap analysis:
  Within-formula : General vs Specific, lang vs lang
  Cross-formula  : matching sets between fh_th_sh and th_fh_sh (Jaccard, enrichment,
                   hypergeometric p-value, Spearman on full BRS vectors)

Output layout
-------------
  output/step4_filtering_Bridge_Routing_Score/{model_short}/{formula_key}/
      {lang}/bridge_scores_{lang}.parquet    # per-head BRS + z-scores
      head_sets.json[l]                      # General + Specific head sets
      overlap_metrics.json[l] / .csv / summary.csv
  output/step4_filtering_Bridge_Routing_Score/{model_short}/
      formula_comparison.json / .csv         # cross-formula overlap

Usage examples
--------------
# GPU jobs (run 3 in parallel, 2 GPUs each):
CUDA_VISIBLE_DEVICES=0,1 python script/step4_filtering_Bridge_Routing_Score.py \\
    --model-short llama31_70 --model meta-llama/Llama-3.1-70B \\
    --langs en ko zh ja es --condition FH

CUDA_VISIBLE_DEVICES=2,3 python script/step4_filtering_Bridge_Routing_Score.py \\
    --model-short llama31_70 --model meta-llama/Llama-3.1-70B \\
    --langs en ko zh ja es --condition TH

CUDA_VISIBLE_DEVICES=4,5 python script/step4_filtering_Bridge_Routing_Score.py \\
    --model-short llama31_70 --model meta-llama/Llama-3.1-70B \\
    --langs en ko zh ja es --condition SH

# CPU-only aggregation (after all three GPU jobs finish):
python script/step4_filtering_Bridge_Routing_Score.py \\
    --model-short llama31_70 --model meta-llama/Llama-3.1-70B \\
    --langs en ko zh ja es --condition aggregate-only
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"

import sys
sys.path.insert(0, str(PROJECT_ROOT / "script"))

from utils.common import save_json, set_seed, setup_logging
from utils.model_utils import load_model_and_tokenizer
from utils.prompt_utils import wrap_prompt
from utils.bridge_utils import (
    all_head_pairs,
    compute_bridge_scores,
    compute_mh_centric_scores,
    compute_overlap_metrics,
    extract_answer_span,
    head_from_flat,
    head_id,
    load_filtered_records,
    load_scores_matrix,
    sample_records,
    save_head_score_df,
    save_scores_matrix,
    top_percent_set,
    zscores,
)
from utils.head_hooks import (
    HeadMaskManager,
    _get_head_dim,
    _get_num_heads,
    _get_num_layers,
    compute_nll_with_grad,
)


# ---------------------------------------------------------------------------
# BRS formula variants
# ---------------------------------------------------------------------------

_FORMULA_FNS: dict[str, Any] = {
    "fh+th-sh": compute_bridge_scores,      # z(FH) + z(TH) - z(SH)
    "th-fh-sh": compute_mh_centric_scores,  # z(TH) - z(FH) - z(SH)
}

# ---------------------------------------------------------------------------
# Condition → prompt / eval key mappings
# ---------------------------------------------------------------------------

_COND_META: dict[str, dict[str, str]] = {
    "FH": {"prompt_key": "first_hop",  "pred_key": "first_hop_pred"},
    "TH": {"prompt_key": "two_hop",    "pred_key": "two_hop_pred"},
    "SH": {"prompt_key": "second_hop", "pred_key": "second_hop_pred"},
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 4: BRS computation and Language-General/Specific head set discovery."
    )
    p.add_argument("--model-short", required=True,
                   help="Short model identifier used in output paths (e.g. llama31_70).")
    p.add_argument("--model", default=None,
                   help="HuggingFace model id or local path. Required for scoring conditions.")
    p.add_argument("--langs", nargs="+", default=["en", "ko", "zh", "ja", "es"],
                   help="Languages to process.")
    p.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "data",
                   help="Root of filtered data; expects "
                        "{model_short}/filtered/{lang}/correct_{model_short}_{lang}.json")
    p.add_argument("--output-root", type=Path,
                   default=PROJECT_ROOT / "output" / "step4_filtering_Bridge_Routing_Score",
                   help="Root for all Step 4 outputs.")
    p.add_argument("--sample-size", type=int, default=600,
                   help="Max number of items per language per condition.")
    p.add_argument("--top-percent", type=float, default=0.13,
                   help="Fraction of heads to include in top-percent sets (e.g. 0.13 = top 13%%).")
    p.add_argument("--specific-percent", type=float, default=0.04,
                   help="Fraction of heads to include in specific sets (e.g. 0.04 = top 4%%).")
    p.add_argument("--gold-mode", default="first_span",
                   choices=["first_span", "first_word", "first_token"],
                   help="How to extract the gold answer span from *_pred fields.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--condition", default=None,
                   choices=["FH", "TH", "SH", "aggregate-only"],
                   help="Which sub-step to run. Omit to run all GPU conditions then aggregate.")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--torch-dtype", default="auto")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Scoring (GPU step)
# ---------------------------------------------------------------------------

def run_scoring_condition(
    condition: str,
    args: argparse.Namespace,
    logger,
    model=None,
    tokenizer=None,
):
    """Compute item × head gradient importance scores for one condition (FH / TH / SH).

    Accepts an already-loaded (model, tokenizer) and returns them so the caller
    reuses ONE model across FH/TH/SH. Loading a fresh 70B copy per condition
    (old behaviour) leaves the previous copy's CUDA memory reserved, so the
    second load gets offloaded to CPU/meta by accelerate and forwards crash
    with 'Tensor on device meta'. The model is loaded lazily, only when at
    least one language still needs scoring."""
    meta = _COND_META[condition]
    prompt_key = meta["prompt_key"]
    pred_key = meta["pred_key"]

    # Determine remaining work before touching the GPU.
    pending_langs = []
    for lang in args.langs:
        out_path = args.output_root / args.model_short / lang / f"scores_{condition}_{lang}.parquet"
        if out_path.exists():
            logger.info("Skipping %s %s — already exists.", condition, lang)
        else:
            pending_langs.append(lang)
    if not pending_langs:
        return model, tokenizer

    if model is None:
        if not args.model:
            raise ValueError("--model is required for scoring conditions.")
        logger.info("Loading model: %s", args.model)
        model, tokenizer = load_model_and_tokenizer(
            args.model,
            device="auto",
            torch_dtype=args.torch_dtype,
            hf_token=args.hf_token,
            trust_remote_code=args.trust_remote_code,
            attn_implementation="eager",
        )

    n_layers = _get_num_layers(model)
    n_heads  = _get_num_heads(model)
    head_dim = _get_head_dim(model)
    logger.info("Model layers=%d  heads=%d  head_dim=%d", n_layers, n_heads, head_dim)

    for lang in pending_langs:
        out_dir = args.output_root / args.model_short / lang
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"scores_{condition}_{lang}.parquet"

        records = load_filtered_records(args.input_root, args.model_short, lang)
        records = sample_records(records, args.sample_size, args.seed)
        logger.info("Sampled %d records for %s/%s", len(records), condition, lang)

        score_rows: list[np.ndarray] = []
        item_ids: list[str] = []
        skipped = 0

        with HeadMaskManager(model, n_layers, n_heads, head_dim) as mask_mgr:
            for record in records:
                raw_pred = record.get("eval", {}).get(pred_key, "")
                gold = extract_answer_span(
                    raw_pred, mode=args.gold_mode,
                    tokenizer=tokenizer if args.gold_mode == "first_token" else None,
                )
                if not gold:
                    skipped += 1
                    continue

                prompt = wrap_prompt(record["prompts"][prompt_key], lang)
                mask_mgr.reset_masks()
                nll, scores = compute_nll_with_grad(model, tokenizer, prompt, gold, mask_mgr)

                if math.isnan(nll):
                    skipped += 1
                    continue

                score_rows.append(scores.reshape(-1))  # [n_layers * n_heads]
                item_ids.append(record["id"])

        logger.info("Scored %d items, skipped %d  (%s/%s)", len(item_ids), skipped, condition, lang)

        if not score_rows:
            logger.warning("No valid scores for %s/%s — skipping save.", condition, lang)
            continue

        matrix = np.stack(score_rows, axis=0).astype(np.float32)
        save_scores_matrix(matrix, item_ids, n_layers, n_heads, out_path)
        logger.info("Saved: %s", out_path)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _load_mean_scores(out_dir: Path, condition: str, lang: str) -> tuple[np.ndarray, int] | None:
    path = out_dir / f"scores_{condition}_{lang}.parquet"
    if not path.exists():
        return None
    matrix, _ = load_scores_matrix(path)
    return matrix.mean(axis=0), matrix.shape[0]


def _build_bridge_df(
    mean_fh: np.ndarray,
    mean_th: np.ndarray,
    mean_sh: np.ndarray,
    n_layers: int,
    n_heads: int,
    model_short: str,
    lang: str,
    brs: np.ndarray,
) -> pd.DataFrame:
    """Build per-head DataFrame with raw scores, z-scores, BRS, and BRS rank."""
    z_fh = zscores(mean_fh)
    z_th = zscores(mean_th)
    z_sh = zscores(mean_sh)
    ranks = (-brs).argsort().argsort() + 1  # rank 1 = highest BRS

    rows = []
    for flat, (l, h) in enumerate(all_head_pairs(n_layers, n_heads)):
        rows.append({
            "model": model_short,
            "lang": lang,
            "layer": l,
            "head": h,
            "head_id": head_id(l, h),
            "flat_head_index": flat,
            "S_FH": float(mean_fh[flat]),
            "S_TH": float(mean_th[flat]),
            "S_SH": float(mean_sh[flat]),
            "z_FH": float(z_fh[flat]),
            "z_TH": float(z_th[flat]),
            "z_SH": float(z_sh[flat]),
            "BRS": float(brs[flat]),
            "rank_brs": int(ranks[flat]),
        })
    return pd.DataFrame(rows)


def _head_set_to_records(
    flat_indices: set[int],
    score_vec: np.ndarray,
    n_heads: int,
) -> list[dict[str, Any]]:
    result = []
    for flat in sorted(flat_indices, key=lambda f: -score_vec[f]):
        l, h = head_from_flat(flat, n_heads)
        result.append({
            "layer": l,
            "head": h,
            "head_id": head_id(l, h),
            "flat_head_index": flat,
            "score": float(score_vec[flat]),
            "rank": int((-score_vec).argsort().argsort()[flat] + 1),
        })
    return result


# ---------------------------------------------------------------------------
# Aggregation (CPU step)
# ---------------------------------------------------------------------------

def run_aggregation(args: argparse.Namespace, logger) -> None:
    """Compute BRS for both formulas, build head sets, and compare formulas."""
    model_dir = args.output_root / args.model_short
    model_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load raw mean scores once (shared across both formulas)
    # ------------------------------------------------------------------
    raw_scores: dict[str, dict[str, np.ndarray]] = {}
    n_layers_ref = n_heads_ref = None

    for lang in args.langs:
        out_dir = model_dir / lang
        fh_res = _load_mean_scores(out_dir, "FH", lang)
        th_res = _load_mean_scores(out_dir, "TH", lang)
        sh_res = _load_mean_scores(out_dir, "SH", lang)

        if any(r is None for r in (fh_res, th_res, sh_res)):
            missing = [c for c, r in zip(["FH", "TH", "SH"], [fh_res, th_res, sh_res]) if r is None]
            logger.warning("Missing files for lang=%s conditions=%s — skipping.", lang, missing)
            continue

        mean_fh, _ = fh_res
        mean_th, _ = th_res
        mean_sh, _ = sh_res

        if n_layers_ref is None:
            cols = [c for c in pd.read_parquet(out_dir / f"scores_FH_{lang}.parquet").columns
                    if c != "item_id"]
            n_layers_ref = max(int(c.split("H")[0][1:]) for c in cols) + 1
            n_heads_ref  = max(int(c.split("H")[1])     for c in cols) + 1
            logger.info("Inferred n_layers=%d n_heads=%d", n_layers_ref, n_heads_ref)

        raw_scores[lang] = {"fh": mean_fh, "th": mean_th, "sh": mean_sh}

    if not raw_scores:
        logger.error("No languages loaded — aborting.")
        return

    n_layers, n_heads = n_layers_ref, n_heads_ref
    n_total = n_layers * n_heads
    langs_present = [l for l in args.langs if l in raw_scores]

    # ------------------------------------------------------------------
    # Per-formula: BRS → head sets → within-formula overlap
    # Collect results for cross-formula comparison afterwards.
    # ------------------------------------------------------------------
    # formula_flat_sets[fk][set_name] = set of flat head indices
    formula_flat_sets: dict[str, dict[str, set[int]]] = {}
    # formula_brs_by_lang[fk][lang] = full BRS vector (n_total,)
    formula_brs_by_lang: dict[str, dict[str, np.ndarray]] = {}
    formula_avg_brs: dict[str, np.ndarray] = {}

    for formula_key, score_fn in _FORMULA_FNS.items():
        formula_dir = model_dir / formula_key
        formula_dir.mkdir(parents=True, exist_ok=True)
        logger.info("=== Formula: %s ===", formula_key)

        # ---- BRS per language ----
        brs_by_lang: dict[str, np.ndarray] = {}
        for lang in langs_present:
            s = raw_scores[lang]
            brs = score_fn(s["fh"], s["th"], s["sh"])
            brs_by_lang[lang] = brs

            lang_dir = formula_dir / lang
            lang_dir.mkdir(parents=True, exist_ok=True)

            df = _build_bridge_df(
                s["fh"], s["th"], s["sh"],
                n_layers, n_heads, args.model_short, lang, brs,
            )
            save_head_score_df(df, lang_dir / f"bridge_scores_{lang}.parquet")
            logger.info("[%s] Saved bridge_scores_%s", formula_key, lang)

        avg_brs = np.mean([brs_by_lang[l] for l in langs_present], axis=0)
        formula_brs_by_lang[formula_key] = brs_by_lang
        formula_avg_brs[formula_key] = avg_brs

        # ---- Head set construction ----
        head_sets: dict[str, list[dict[str, Any]]] = {}
        flat_sets: dict[str, set[int]] = {}

        def _add(name: str, indices: set[int], scores: np.ndarray) -> None:
            flat_sets[name] = indices
            head_sets[name] = _head_set_to_records(indices, scores, n_heads)

        # Per-language top-percent
        bridge_top_sets: dict[str, set[int]] = {}
        for lang in langs_present:
            brs = brs_by_lang[lang]
            top_set = top_percent_set(brs, args.top_percent)
            bridge_top_sets[lang] = top_set
            _add(f"H_Bridge_{lang.upper()}", top_set, brs)

        # Language-General: strict intersection
        if len(bridge_top_sets) >= 2:
            general: set[int] = set.intersection(*bridge_top_sets.values())
        elif len(bridge_top_sets) == 1:
            general = next(iter(bridge_top_sets.values())).copy()
        else:
            general = set()
        _add("H_Bridge_General", general, avg_brs)
        logger.info("[%s] H_Bridge_General: %d heads (top_percent=%.3f, n_langs=%d)",
                    formula_key, len(general), args.top_percent, len(langs_present))

        # Language-Specific: top-percent after masking out General
        for lang in langs_present:
            brs = brs_by_lang[lang]
            masked = brs.copy()
            for h in general:
                masked[h] = -np.inf
            specific = top_percent_set(masked, args.specific_percent)  # use all remaining heads to define specific set
            _add(f"H_Bridge_Specific_{lang.upper()}", specific, brs)
            logger.info("[%s] H_Bridge_Specific_%s: %d heads (excluded %d general)",
                        formula_key, lang.upper(), len(specific), len(general))

        formula_flat_sets[formula_key] = dict(flat_sets)

        # ---- Save head sets ----
        save_json(head_sets, formula_dir / "head_sets.json")
        with (formula_dir / "head_sets.jsonl").open("w", encoding="utf-8") as f:
            for name, recs in head_sets.items():
                f.write(json.dumps({"set_name": name, "heads": recs}, ensure_ascii=False) + "\n")
        logger.info("[%s] Saved head_sets (%d sets)", formula_key, len(head_sets))

        # ---- Within-formula overlap metrics ----
        def _sv(sname: str) -> np.ndarray:
            if sname == "H_Bridge_General":
                return avg_brs
            if sname.startswith("H_Bridge_Specific_"):
                lc = sname.removeprefix("H_Bridge_Specific_").lower()
                return brs_by_lang.get(lc, np.zeros(n_total))
            lc = sname.removeprefix("H_Bridge_").lower()
            return brs_by_lang.get(lc, np.zeros(n_total))

        pairs: list[tuple[str, str]] = []
        for lang in langs_present:
            pairs.append(("H_Bridge_General", f"H_Bridge_Specific_{lang.upper()}"))
        for i, la in enumerate(langs_present):
            for lb in langs_present[i + 1:]:
                pairs.append((f"H_Bridge_{la.upper()}", f"H_Bridge_{lb.upper()}"))

        overlap_records = [
            compute_overlap_metrics(na, flat_sets[na], _sv(na), nb, flat_sets[nb], _sv(nb), n_total)
            for na, nb in pairs
            if flat_sets.get(na) and flat_sets.get(nb)
        ]

        save_json(overlap_records, formula_dir / "overlap_metrics.json")
        if overlap_records:
            df_ov = pd.DataFrame(overlap_records)
            df_ov.to_csv(formula_dir / "overlap_metrics.csv", index=False)
            summary_cols = ["set_a", "set_b", "intersection", "jaccard",
                            "enrichment", "hypergeometric_pvalue", "spearman"]
            df_ov[summary_cols].sort_values("jaccard", ascending=False).to_csv(
                formula_dir / "summary.csv", index=False
            )
        logger.info("[%s] Saved overlap_metrics", formula_key)

    # ------------------------------------------------------------------
    # Cross-formula comparison (fh_th_sh vs th_fh_sh)
    # Compares matching head sets between the two formula variants.
    # ------------------------------------------------------------------
    if len(formula_flat_sets) == 2:
        logger.info("=== Cross-formula comparison ===")
        fk_a, fk_b = list(_FORMULA_FNS.keys())
        sets_a  = formula_flat_sets[fk_a]
        sets_b  = formula_flat_sets[fk_b]
        brs_a   = formula_brs_by_lang[fk_a]
        brs_b   = formula_brs_by_lang[fk_b]
        avg_a   = formula_avg_brs[fk_a]
        avg_b   = formula_avg_brs[fk_b]

        def _sv_cross(sname: str, fk: str) -> np.ndarray:
            brs_map = formula_brs_by_lang[fk]
            avg     = formula_avg_brs[fk]
            if sname == "H_Bridge_General":
                return avg
            if sname.startswith("H_Bridge_Specific_"):
                lc = sname.removeprefix("H_Bridge_Specific_").lower()
                return brs_map.get(lc, np.zeros(n_total))
            lc = sname.removeprefix("H_Bridge_").lower()
            return brs_map.get(lc, np.zeros(n_total))

        # Compare: General, Specific per lang, per-lang bridge — same set name, different formula
        cross_set_names = ["H_Bridge_General"]
        for lang in langs_present:
            cross_set_names.append(f"H_Bridge_Specific_{lang.upper()}")
        for lang in langs_present:
            cross_set_names.append(f"H_Bridge_{lang.upper()}")

        cross_records: list[dict] = []
        for sname in cross_set_names:
            sa = sets_a.get(sname, set())
            sb = sets_b.get(sname, set())
            if not sa or not sb:
                continue
            cross_records.append(compute_overlap_metrics(
                f"{sname} [{fk_a}]", sa, _sv_cross(sname, fk_a),
                f"{sname} [{fk_b}]", sb, _sv_cross(sname, fk_b),
                n_total,
            ))

        save_json(cross_records, model_dir / "formula_comparison.json")
        if cross_records:
            df_cross = pd.DataFrame(cross_records)
            df_cross.to_csv(model_dir / "formula_comparison.csv", index=False)
            logger.info("Cross-formula comparison: %d pairs — saved formula_comparison.*",
                        len(cross_records))
            logger.info("\n%s", df_cross[["set_a", "set_b", "intersection", "jaccard",
                                          "spearman"]].to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    logger = setup_logging(f"step4_BRS_{args.model_short}", LOG_DIR)
    logger.info("Step 4 — condition=%s  model=%s  langs=%s",
                args.condition, args.model_short, args.langs)

    if args.condition in ("FH", "TH", "SH"):
        run_scoring_condition(args.condition, args, logger)
    elif args.condition == "aggregate-only":
        run_aggregation(args, logger)
    else:
        # Single-machine mode: all three conditions then aggregate.
        # One model instance is loaded lazily and shared across conditions --
        # re-loading per condition would double-book GPU memory and push the
        # second copy onto CPU/meta (accelerate offload) on 80GB cards.
        model = tokenizer = None
        for cond in ("FH", "TH", "SH"):
            logger.info("=== Scoring: %s ===", cond)
            model, tokenizer = run_scoring_condition(cond, args, logger, model, tokenizer)
        logger.info("=== Aggregation ===")
        run_aggregation(args, logger)

    logger.info("Step 4 complete.")


if __name__ == "__main__":
    main()
