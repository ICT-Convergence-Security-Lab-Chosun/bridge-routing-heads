"""Extra 3: Layer-matched random head ablation baseline.

For each bridge head set (general / language-specific), instead of drawing a
random ablation pool from the *entire* model, sample replacement heads from
the **same layers** as the bridge heads—excluding the bridge heads themselves.
Pool size k always equals the number of bridge heads being tested.

Motivation
----------
Step 6 Stage 2 uses a uniform-random baseline that can accidentally select
heads from layers very different from the bridge head layers. A fairer
control ablates the same number of heads drawn proportionally from exactly
the same layers, removing any layer-depth confound.

Protocol
--------
1.  Calibrate mean head outputs using Step-6-identical calibration:
    50 correct two-hop prompts × (en + ko/ja/zh/es) → per-head scalar means.

2.  For each test language, load correct two-hop records (--ablation-sample-size,
    default 100).  Build valid items the same way as Step 6 (NLL-base filter).

3.  For **general** and **specific** head sets separately:
    a.  Identify the set of bridge heads B ⊂ {(layer, head)}.
    b.  Build the layer-matched pool P:
          For each (L, H) in B, eligible candidates = {(L, h) : h ≠ H, (L,h) ∉ B}
          Sample without replacement (within each layer) to produce |B| heads.
          If a layer has too few non-bridge candidates, fall back to all
          non-bridge heads in that layer (sampling with replacement if needed).
    c.  Repeat --random-repeat times (default 20), re-drawing P each time.
    d.  Mean-ablate the random pool and record ΔNLL.

4.  Aggregate and save results to output/extra/extra3_layer_matched_ablation/.

Usage
-----
    python script/extra/extra3_layer_matched_ablation.py \\
        --model-short qwen25_72 --model Qwen/Qwen2.5-72B \\
        --langs ko zh ja es

    # Jaccard-style dry-run (no model)
    python script/extra/extra3_layer_matched_ablation.py \\
        --model-short qwen25_72 --model dummy \\
        --langs ko zh ja es --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"

import sys
sys.path.insert(0, str(PROJECT_ROOT / "script"))

from utils.common import load_json, save_json, set_seed, setup_logging
from utils.model_utils import load_model_and_tokenizer
from utils.prompt_utils import wrap_prompt
from utils.bridge_utils import extract_answer_span, load_filtered_records, sample_records
from utils.head_hooks import (
    HeadMaskManager,
    _get_head_dim,
    _get_num_heads,
    _get_num_layers,
    collect_mean_head_outputs,
    compute_nll_eval,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extra 3: Layer-matched random head ablation baseline."
    )
    p.add_argument("--model-short", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--langs", nargs="+", default=["ko", "zh", "ja", "es"],
                   help="Target languages for ablation.")
    p.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--step5-root", type=Path,
                   default=PROJECT_ROOT / "output" / "step5_BridgeHead_Ablation_Patchscopes",
                   help="Root of Step 5 output (contains {model_short}/final_bridge_heads.json).")
    p.add_argument("--output-root", type=Path,
                   default=PROJECT_ROOT / "output" / "extra" / "extra3_layer_matched_ablation")

    p.add_argument("--ablation-sample-size", type=int, default=100,
                   help="Correct records per language for ablation test (default: 100).")
    p.add_argument("--calibration-size", type=int, default=50,
                   help="Correct records per language for mean calibration (default: 50).")
    p.add_argument("--random-repeat", type=int, default=20,
                   help="How many times to resample the layer-matched random pool (default: 20).")

    p.add_argument("--gold-mode", default="first_span",
                   choices=["first_span", "first_word", "first_token"],
                   help="Gold extraction mode for ablation NLL from eval.two_hop_pred.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hf-token", default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--torch-dtype", default="auto")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip model load; only print head-set stats.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Bridge-head loading  (same as step6)
# ---------------------------------------------------------------------------

def load_final_bridge_heads(step5_dir: Path, model_short: str) -> dict[str, Any]:
    path = step5_dir / model_short / "final_bridge_heads.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data["final_bridge_heads"]
    return {
        "general": [(int(r[0]), int(r[1])) for r in raw["general"]],
        "specific": {
            lang: [(int(r[0]), int(r[1])) for r in heads]
            for lang, heads in raw.get("specific", {}).items()
        },
    }


# ---------------------------------------------------------------------------
# Layer-matched random sampling
# ---------------------------------------------------------------------------

def sample_layer_matched_heads(
    bridge_heads: list[tuple[int, int]],
    n_heads: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    """Sample k = len(bridge_heads) heads, one per bridge head, from the same layer.

    For each bridge head (L, H), eligible candidates are all heads in layer L
    that are NOT in the bridge set.

    Fallback (layer fully covered by bridge heads):
        Pick from OTHER layers that contain bridge heads but still have at
        least one non-bridge head available.  This preserves the "bridge-head
        layer" constraint while avoiding the degenerate same-head selection.

    Final fallback (all bridge-head layers fully covered):
        Pick from the entire bridge-head layer pool ignoring exclusions
        (extremely degenerate; included only for correctness).

    Always returns exactly k heads.
    """
    bridge_set = set(bridge_heads)
    # Precompute per-layer bridge heads for quick exclusion
    bridge_by_layer: dict[int, set[int]] = {}
    for (l, h) in bridge_set:
        bridge_by_layer.setdefault(l, set()).add(h)

    # Non-bridge candidates per layer (only layers that contain bridge heads)
    candidates_by_layer: dict[int, list[int]] = {
        l: [h for h in range(n_heads) if h not in bh_set]
        for l, bh_set in bridge_by_layer.items()
    }

    # Cross-layer fallback pool: (layer, head) pairs from bridge-head layers
    # that still have at least one non-bridge head.
    fallback_pool: list[tuple[int, int]] = [
        (l, h)
        for l, cands in candidates_by_layer.items()
        for h in cands
    ]

    result: list[tuple[int, int]] = []
    for (layer, _head) in bridge_heads:
        cands = candidates_by_layer.get(layer, [])
        if cands:
            result.append((layer, rng.choice(cands)))
        elif fallback_pool:
            # Layer fully covered: pick from another bridge-head layer that
            # still has non-bridge heads remaining.
            result.append(rng.choice(fallback_pool))
        else:
            # Truly degenerate: every bridge-head layer is fully covered.
            # Pick any head from any bridge-head layer (last resort).
            any_pool = [(l, h) for l, bh in bridge_by_layer.items()
                        for h in range(n_heads)]
            result.append(rng.choice(any_pool))
    return result


# ---------------------------------------------------------------------------
# Calibration  (identical to Step 6 Stage 2)
# ---------------------------------------------------------------------------

def build_calibration_means(
    args: argparse.Namespace,
    model,
    tokenizer,
    logger,
) -> dict[tuple[int, int], float]:
    n_layers = _get_num_layers(model)
    n_heads = _get_num_heads(model)
    head_dim = _get_head_dim(model)

    calib_langs = ["en", "ko", "ja", "zh", "es"]
    prompts: list[str] = []
    for clang in calib_langs:
        try:
            records = load_filtered_records(args.input_root, args.model_short, clang)
        except FileNotFoundError:
            logger.warning("Calibration: no correct records for lang=%s -- skipping.", clang)
            continue
        for rec in sample_records(records, args.calibration_size, args.seed):
            prompts.append(wrap_prompt(rec["prompts"]["two_hop"], rec["lang"]))

    logger.info(
        "Calibration: %d prompts from %d langs (target %d × %d = %d).",
        len(prompts), len(calib_langs),
        len(calib_langs), args.calibration_size, len(calib_langs) * args.calibration_size,
    )
    mean_values = collect_mean_head_outputs(model, tokenizer, prompts, n_layers, n_heads, head_dim)
    logger.info("Mean head outputs collected.")
    return mean_values


# ---------------------------------------------------------------------------
# Valid-item collection  (identical to Step 6 Stage 2)
# ---------------------------------------------------------------------------

def collect_valid_items(
    model,
    tokenizer,
    records: list[dict],
    gold_mode: str,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for record in records:
        raw_pred = record.get("eval", {}).get("two_hop_pred", "")
        gold = extract_answer_span(raw_pred, mode=gold_mode, tokenizer=tokenizer)
        if not gold:
            continue
        prompt = wrap_prompt(record["prompts"]["two_hop"], record["lang"])
        nll_base = compute_nll_eval(model, tokenizer, prompt, gold)
        if math.isnan(nll_base):
            continue
        items.append({"id": record["id"], "prompt": prompt, "gold": gold, "nll_base": nll_base})
    return items


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    arr = np.array(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": len(arr)}


def bootstrap_ci(
    values: list[float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    arr = np.array(values, dtype=np.float64)
    rng_np = np.random.default_rng(seed)
    boot_means = np.array([
        rng_np.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1.0 - ci) / 2.0
    return (
        float(np.percentile(boot_means, alpha * 100)),
        float(np.percentile(boot_means, (1.0 - alpha) * 100)),
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _save_results(records: list[dict], path_stem: Path) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    if records:
        pd.DataFrame(records).to_csv(path_stem.with_suffix(".csv"), index=False)
    with path_stem.with_suffix(".jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Core ablation loop
# ---------------------------------------------------------------------------

def run_layer_matched_ablation_for_head_set(
    *,
    lang: str,
    head_set_type: str,               # "general" or "specific"
    bridge_heads: list[tuple[int, int]],
    items: list[dict[str, Any]],
    mean_values: dict[tuple[int, int], float],
    mask_mgr: HeadMaskManager,
    n_heads: int,
    random_repeat: int,
    seed: int,
    model,
    tokenizer,
    logger,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run layer-matched random ablation for one head set type on one language.

    Returns
    -------
    (summary_rows, per_item_rows)
        summary_rows  : one row per repeat (aggregate stats)
        per_item_rows : per-item NLL for all repeats combined
    """
    k = len(bridge_heads)
    base_nlls = [item["nll_base"] for item in items]
    base_nll_mean = float(np.mean(base_nlls))

    rng = random.Random(seed)
    summary_rows: list[dict[str, Any]] = []
    per_item_rows: list[dict[str, Any]] = []
    all_deltas: list[float] = []

    for repeat_i in range(random_repeat):
        rand_pool = sample_layer_matched_heads(bridge_heads, n_heads, rng)

        repeat_nlls: list[float] = []
        for item in items:
            mask_mgr.reset_masks()
            mask_mgr.apply_head_set_ablation(rand_pool, mean_values)
            nll_a = compute_nll_eval(model, tokenizer, item["prompt"], item["gold"])
            mask_mgr.reset_masks()
            delta = nll_a - item["nll_base"]
            repeat_nlls.append(nll_a)
            all_deltas.append(delta)
            per_item_rows.append({
                "lang": lang,
                "head_set_type": head_set_type,
                "repeat": repeat_i,
                "id": item["id"],
                "nll_base": item["nll_base"],
                "nll_ablated": nll_a,
                "delta": delta,
                "rand_pool": str(rand_pool),
            })

        repeat_delta = [x - b for x, b in zip(repeat_nlls, base_nlls)]
        r_stats = _summary_stats(repeat_delta)
        summary_rows.append({
            "lang": lang,
            "head_set_type": head_set_type,
            "repeat": repeat_i,
            "k": k,
            "sample_size": len(items),
            "base_nll_mean": base_nll_mean,
            "ablated_nll_mean": float(np.mean(repeat_nlls)),
            "delta_nll_mean": r_stats["mean"],
            "delta_nll_std": r_stats["std"],
        })

    # Aggregate across all repeats
    agg_stats = _summary_stats(all_deltas)
    agg_ci_lo, agg_ci_hi = bootstrap_ci(all_deltas, seed=seed)
    logger.info(
        "layer-matched-rand  lang=%-4s  type=%-8s  k=%d  "
        "delta=%.4f [%.4f, %.4f]  (n_deltas=%d over %d repeats)",
        lang, head_set_type, k,
        agg_stats["mean"], agg_ci_lo, agg_ci_hi,
        len(all_deltas), random_repeat,
    )

    return summary_rows, per_item_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    logger = setup_logging(
        f"extra3_layer_matched_ablation_{args.model_short}", LOG_DIR
    )
    logger.info(
        "Extra 3 -- model=%s  langs=%s  random_repeat=%d",
        args.model_short, args.langs, args.random_repeat,
    )

    bridge_heads = load_final_bridge_heads(args.step5_root, args.model_short)
    logger.info(
        "Loaded bridge heads: general=%d  specific=%s",
        len(bridge_heads["general"]),
        {k: len(v) for k, v in bridge_heads["specific"].items()},
    )

    if args.dry_run:
        logger.info("--dry-run: skipping model load and ablation.")
        for lang in args.langs:
            for htype in ("general", "specific"):
                bh = (
                    bridge_heads["general"]
                    if htype == "general"
                    else bridge_heads["specific"].get(lang, [])
                )
                logger.info(
                    "  lang=%-4s  type=%-8s  k=%d  layers=%s",
                    lang, htype, len(bh), sorted({l for (l, _) in bh}),
                )
        return

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
    n_heads = _get_num_heads(model)
    head_dim = _get_head_dim(model)

    # Calibration (same as Step 6 Stage 2)
    mean_values = build_calibration_means(args, model, tokenizer, logger)

    all_summary: list[dict[str, Any]] = []
    all_per_item: list[dict[str, Any]] = []

    out_model_dir = args.output_root / args.model_short
    out_model_dir.mkdir(parents=True, exist_ok=True)

    with HeadMaskManager(model, n_layers, n_heads, head_dim) as mask_mgr:
        for lang in args.langs:
            try:
                records = load_filtered_records(args.input_root, args.model_short, lang)
            except FileNotFoundError:
                logger.warning("Correct records not found for lang=%s -- skipping.", lang)
                continue
            records = sample_records(records, args.ablation_sample_size, args.seed)
            items = collect_valid_items(model, tokenizer, records, args.gold_mode)
            logger.info(
                "lang=%s: %d/%d valid items after NLL-base filter.",
                lang, len(items), len(records),
            )
            if not items:
                continue

            lang_summary: list[dict[str, Any]] = []
            lang_per_item: list[dict[str, Any]] = []

            for head_set_type in ("general", "specific"):
                bridge_set = (
                    bridge_heads["general"]
                    if head_set_type == "general"
                    else bridge_heads["specific"].get(lang, [])
                )
                if not bridge_set:
                    logger.warning(
                        "Empty bridge head set for lang=%s type=%s -- skipping.",
                        lang, head_set_type,
                    )
                    continue

                logger.info(
                    "=== Layer-matched ablation: lang=%s  type=%s  k=%d ===",
                    lang, head_set_type, len(bridge_set),
                )
                s_rows, p_rows = run_layer_matched_ablation_for_head_set(
                    lang=lang,
                    head_set_type=head_set_type,
                    bridge_heads=bridge_set,
                    items=items,
                    mean_values=mean_values,
                    mask_mgr=mask_mgr,
                    n_heads=n_heads,
                    random_repeat=args.random_repeat,
                    seed=args.seed,
                    model=model,
                    tokenizer=tokenizer,
                    logger=logger,
                )
                lang_summary.extend(s_rows)
                lang_per_item.extend(p_rows)

            # Per-language outputs
            lang_dir = out_model_dir / lang
            lang_dir.mkdir(parents=True, exist_ok=True)
            _save_results(lang_summary, lang_dir / "layer_matched_summary")
            _save_results(lang_per_item, lang_dir / "layer_matched_per_item")

            all_summary.extend(lang_summary)
            all_per_item.extend(lang_per_item)

    # Aggregate across all languages
    _save_results(all_summary, out_model_dir / "layer_matched_summary_all_langs")
    _save_results(all_per_item, out_model_dir / "layer_matched_per_item_all_langs")

    # Compute aggregate stats per (lang, head_set_type) and save as JSON summary
    agg_rows: list[dict[str, Any]] = []
    if all_summary:
        df = pd.DataFrame(all_summary)
        for (lang, htype), grp in df.groupby(["lang", "head_set_type"]):
            deltas = []
            for _, row in grp.iterrows():
                # Reconstruct mean delta per repeat already computed
                deltas.append(row["delta_nll_mean"])
            agg_stats = _summary_stats(deltas)
            agg_ci_lo, agg_ci_hi = bootstrap_ci(deltas, seed=args.seed)
            agg_rows.append({
                "model": args.model_short,
                "lang": lang,
                "head_set_type": htype,
                "baseline_type": "layer_matched_random",
                "intervention": "mean_ablation",
                "k": int(grp["k"].iloc[0]),
                "sample_size": int(grp["sample_size"].iloc[0]),
                "n_repeats": args.random_repeat,
                "base_nll_mean": float(grp["base_nll_mean"].iloc[0]),
                "delta_nll_mean": agg_stats["mean"],
                "delta_nll_std": agg_stats["std"],
                "delta_nll_ci_lo": agg_ci_lo,
                "delta_nll_ci_hi": agg_ci_hi,
            })

    save_json(
        {
            "model": args.model_short,
            "langs": args.langs,
            "random_repeat": args.random_repeat,
            "ablation_sample_size": args.ablation_sample_size,
            "calibration_size": args.calibration_size,
            "n_general_heads": len(bridge_heads["general"]),
            "n_specific_heads": {k: len(v) for k, v in bridge_heads["specific"].items()},
            "aggregate": agg_rows,
        },
        out_model_dir / "extra3_summary.json",
    )

    if agg_rows:
        pd.DataFrame(agg_rows).to_csv(out_model_dir / "extra3_aggregate.csv", index=False)

    logger.info(
        "Extra 3 complete. Results -> %s  (langs=%d, agg_rows=%d)",
        out_model_dir, len(args.langs), len(agg_rows),
    )


if __name__ == "__main__":
    main()
