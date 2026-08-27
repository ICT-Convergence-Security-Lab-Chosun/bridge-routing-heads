"""Step 6: Bridge Head Set validation via causal interventions.

Uses final_bridge_heads.json from Step 5 (general + language-specific head sets).

Stages
------
Stage 1 -- Jaccard overlap analysis
    All C(n,2) pairs among {general, ko, zh, ja, es} head sets.
    No model needed.

Stage 2 -- Mean ablation
    Calibration: 50 correct two-hop prompts x (en + ko/ja/zh/es) -> mean head outputs.
    Gold for NLL: clean span extracted from eval.two_hop_pred.
    Test: 100 correct records per language, ablate general set and specific set separately.
    Compare against layer-matched random head set of equal size (20 repeats):
    for each bridge head (L, H), the random pool draws from the same layer L
    excluding bridge heads, falling back to other bridge-head layers if needed.

Stage 3 -- Scaling amplification
    Language-specific heads only (per lang).
    Data: incorrect_{model}_{lang}.json (already two-hop incorrect).
    Alpha in {0.25, 0.5, 1.0}; report amplified accuracy only.

Stage 4 -- Cross-lingual transfer accuracy
    General heads only; amplify during target-language inference.
    Data: en_correct_{lang}_incorrect_{model}.json.
    base_acc = 0 by construction; report amplified accuracy and delta.

Stages 3 & 4 support ``--head-selection {bridge, random}``: "bridge" (default)
amplifies the actual Bridge Head set; "random" amplifies a same-size set of
uniformly random (layer, head) pairs instead, to see how amplification
behaves for a non-bridge control group. Random-selection outputs are written
to separate files (suffixed ``_random_heads``) so they never overwrite the
original bridge-head results.

Usage examples
--------------
# All stages
python script/step6_Bridge_Head_validation.py \
    --model-short qwen25_72 --model Qwen/Qwen2.5-72B \
    --langs ko zh ja es

# Jaccard only (no model load)
python script/step6_Bridge_Head_validation.py \
    --model-short qwen25_72 --model dummy \
    --langs ko zh ja es --stage jaccard

# Scaling only, max 100 incorrect records per language
python script/step6_Bridge_Head_validation.py \
    --model-short qwen25_72 --model Qwen/Qwen2.5-72B \
    --langs ko zh ja es --stage scaling --scaling-max-sample 100

# Stage 3/4 with a random (non-bridge) head group of the same size, for comparison
python script/step6_Bridge_Head_validation.py \
    --model-short qwen25_72 --model Qwen/Qwen2.5-72B \
    --langs ko zh ja es --stage all --head-selection random
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"

import sys
sys.path.insert(0, str(PROJECT_ROOT / "script"))

from utils.common import load_json, save_json, set_seed, setup_logging
from utils.model_utils import check_answer, load_model_and_tokenizer, predict_next_tokens
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
        description="Step 6: Validate Bridge Head sets via causal interventions."
    )
    p.add_argument("--model-short", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--langs", nargs="+", default=["ko", "zh", "ja", "es"],
                   help="Target languages for validation. Calibration always uses en, ko, ja, zh, es.")
    p.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--step5-root", type=Path,
                   default=PROJECT_ROOT / "output" / "step5_BridgeHead_Ablation_Patchscopes",
                   help="Root of Step 5 output (contains {model_short}/final_bridge_heads.json).")
    p.add_argument("--output-root", type=Path,
                   default=PROJECT_ROOT / "output" / "step6_Bridge_Head_validation")
    p.add_argument("--stage", default="all",
                   choices=["jaccard", "ablation", "scaling", "transfer", "all"],
                   help="Which stage to run (default: all). 'jaccard' skips model load.")

    # Stage 2: mean ablation
    p.add_argument("--ablation-sample-size", type=int, default=100,
                   help="Correct records per language for ablation test (default: 100).")
    p.add_argument("--calibration-size", type=int, default=50,
                   help="Correct records per language for mean calibration (default: 50).")
    p.add_argument("--random-repeat", type=int, default=20,
                   help="Random-baseline repeats for ablation (default: 20).")

    # Stage 3 & 4: scaling / transfer
    p.add_argument("--scaling-max-sample", type=int, default=0,
                   help="Max records for scaling and transfer (0 = use all; falls back to all "
                        "when data < N).")
    p.add_argument("--alpha-list", nargs="+", type=float, default=[0.25, 0.5, 1.0],
                   help="Alpha values for scaling amplification.")
    p.add_argument("--head-selection", default="bridge",
                   choices=["bridge", "random"],
                   help="Head group used for amplification in Stage 3 & 4 (default: bridge). "
                        "'bridge' amplifies the actual Bridge Head set; 'random' amplifies a "
                        "same-size set of uniformly random (layer, head) pairs instead, drawn "
                        "fresh per lang/target. Random-mode outputs are saved to separate files "
                        "and never overwrite 'bridge' results.")

    # Stage 4: cross-lingual transfer
    p.add_argument("--cross-lingual-root", type=Path,
                   default=PROJECT_ROOT / "data",
                   help="Root containing filtered/cross_lang/ JSON files.")

    # Generation / misc
    p.add_argument("--n-generate-tokens", type=int, default=10,
                   help="Max new tokens for greedy generation in transfer/scaling.")
    p.add_argument("--gold-mode", default="first_span",
                   choices=["first_span", "first_word", "first_token"],
                   help="Gold extraction mode for ablation NLL (kept for compatibility).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hf-token", default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--torch-dtype", default="auto")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Bridge-head loading
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


def sample_layer_matched_heads(
    bridge_heads: list[tuple[int, int]],
    n_heads: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    """Sample k = len(bridge_heads) heads, one per bridge head, from the same layer.

    For each bridge head (L, H), eligible candidates are all heads in layer L
    that are NOT in the bridge set.

    Fallback (layer fully covered by bridge heads):
        Pick from other layers that contain bridge heads but still have at
        least one non-bridge head available.

    Final fallback (all bridge-head layers fully covered):
        Pick from the entire bridge-head layer pool ignoring exclusions.
    """
    bridge_set = set(bridge_heads)
    bridge_by_layer: dict[int, set[int]] = {}
    for (l, h) in bridge_set:
        bridge_by_layer.setdefault(l, set()).add(h)

    candidates_by_layer: dict[int, list[int]] = {
        l: [h for h in range(n_heads) if h not in bh_set]
        for l, bh_set in bridge_by_layer.items()
    }

    # Cross-layer fallback pool: pairs from bridge-head layers that still
    # have at least one non-bridge head.
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
            result.append(rng.choice(fallback_pool))
        else:
            any_pool = [(l, h) for l, bh in bridge_by_layer.items()
                        for h in range(n_heads)]
            result.append(rng.choice(any_pool))
    return result


def sample_random_head_set(
    head_set: list[tuple[int, int]],
    n_layers: int,
    n_heads: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    """Sample k = len(head_set) uniformly random (layer, head) pairs.

    Unlike ``sample_layer_matched_heads`` (Stage 2's layer-matched control),
    this draws from the entire model with no layer constraint -- used as a
    plain random-head control for Stage 3/4 amplification. The original
    ``head_set`` itself is excluded from the draw pool so the control is
    guaranteed to be a different group.
    """
    k = len(head_set)
    exclude = set(head_set)
    pool = [(l, h) for l in range(n_layers) for h in range(n_heads) if (l, h) not in exclude]
    if len(pool) < k:
        pool = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    return rng.sample(pool, k)


# ---------------------------------------------------------------------------
# Data-loading helpers
# ---------------------------------------------------------------------------

def load_incorrect_records(input_root: Path, model_short: str, lang: str) -> list[dict]:
    path = input_root / model_short / "filtered" / lang / f"incorrect_{model_short}_{lang}.json"
    return load_json(path)


def _apply_scaling_sample(records: list[dict], max_sample: int, seed: int) -> list[dict]:
    if max_sample <= 0 or max_sample >= len(records):
        return list(records)
    return sample_records(records, max_sample, seed)


def _gold_labels(record: dict) -> list[str]:
    labels: list[str] = []
    e3 = record.get("e3", {})
    if isinstance(e3, dict):
        for key in ("label", "label_en"):
            v = e3.get(key)
            if v and str(v).strip():
                labels.append(str(v).strip())
    for key in ("e3_label", "e3_label_en"):
        v = record.get(key)
        if v and str(v).strip() and str(v).strip() not in labels:
            labels.append(str(v).strip())
    return labels


def _save_results(records: list[dict], path_stem: Path) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    if records:
        pd.DataFrame(records).to_csv(path_stem.with_suffix(".csv"), index=False)
    with path_stem.with_suffix(".jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Stage 1: Jaccard Overlap Analysis
# ---------------------------------------------------------------------------

def _jaccard(a: set, b: set) -> float:
    u = len(a | b)
    return float("nan") if u == 0 else len(a & b) / u


def run_jaccard_analysis(
    bridge_heads: dict[str, Any],
    langs: list[str],
    out_dir: Path,
    logger,
) -> list[dict]:
    named_sets: dict[str, set[tuple[int, int]]] = {
        "general": set(bridge_heads["general"])
    }
    for lang in langs:
        if lang in bridge_heads["specific"] and bridge_heads["specific"][lang]:
            named_sets[lang] = set(bridge_heads["specific"][lang])
        else:
            logger.warning("No specific heads for lang=%s -- omitting from Jaccard.", lang)

    rows: list[dict] = []
    for name_a, name_b in itertools.combinations(named_sets.keys(), 2):
        a, b = named_sets[name_a], named_sets[name_b]
        j = _jaccard(a, b)
        rows.append({
            "set_a": name_a,
            "set_b": name_b,
            "size_a": len(a),
            "size_b": len(b),
            "intersection": len(a & b),
            "union": len(a | b),
            "jaccard": j,
        })
        logger.info(
            "Jaccard  %-10s x %-10s  |A&B|=%d  |AuB|=%d  J=%.4f",
            name_a, name_b, len(a & b), len(a | b), j,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    _save_results(rows, out_dir / "jaccard_overlap")
    logger.info("Jaccard analysis complete: %d pairs -> %s", len(rows), out_dir / "jaccard_overlap.csv")
    return rows


# ---------------------------------------------------------------------------
# Stage 2: Mean Ablation
# ---------------------------------------------------------------------------

def _collect_valid_items_ablation(
    model,
    tokenizer,
    records: list[dict],
    gold_mode: str = "first_span",
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
    """Bootstrap percentile CI for the mean. Returns (ci_lo, ci_hi)."""
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


def _ablation_nlls(
    model,
    tokenizer,
    items: list[dict[str, Any]],
    head_set: list[tuple[int, int]],
    mean_values: dict[tuple[int, int], float],
    mask_mgr: HeadMaskManager,
) -> tuple[list[float], list[dict[str, Any]]]:
    """Ablate head_set on all items. Returns (ablated_nlls, per_item_records).

    per_item_records entries: {id, nll_base, nll_ablated, delta}
    """
    nlls: list[float] = []
    per_item: list[dict[str, Any]] = []
    for item in items:
        mask_mgr.reset_masks()
        mask_mgr.apply_head_set_ablation(head_set, mean_values)
        nll_a = compute_nll_eval(model, tokenizer, item["prompt"], item["gold"])
        nlls.append(nll_a)
        per_item.append({
            "id": item["id"],
            "nll_base": item["nll_base"],
            "nll_ablated": nll_a,
            "delta": nll_a - item["nll_base"],
        })
    mask_mgr.reset_masks()
    return nlls, per_item


def run_mean_ablation(
    args: argparse.Namespace,
    model,
    tokenizer,
    bridge_heads: dict[str, Any],
    out_dir: Path,
    logger,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    n_layers = _get_num_layers(model)
    n_heads = _get_num_heads(model)
    head_dim = _get_head_dim(model)

    # Calibration: gather mean head outputs from correct two-hop prompts
    # 50 per lang x (en + ko/ja/zh/es) = up to 250 prompts.
    # Keep this fixed across validation runs so mean ablation uses the same
    # multilingual calibration distribution even when --langs is a subset.
    calib_langs = ["en", "ko", "ja", "zh", "es"]
    calib_prompts: list[str] = []
    for clang in calib_langs:
        try:
            calib_records = load_filtered_records(args.input_root, args.model_short, clang)
        except FileNotFoundError:
            logger.warning("Calibration: correct records not found for lang=%s -- skipping.", clang)
            continue
        calib_sample = sample_records(calib_records, args.calibration_size, args.seed)
        for rec in calib_sample:
            calib_prompts.append(wrap_prompt(rec["prompts"]["two_hop"], rec["lang"]))

    logger.info(
        "Calibration: %d prompts from %d langs (target %d x %d = %d).",
        len(calib_prompts), len(calib_langs),
        len(calib_langs), args.calibration_size, len(calib_langs) * args.calibration_size,
    )
    mean_values = collect_mean_head_outputs(
        model, tokenizer, calib_prompts, n_layers, n_heads, head_dim
    )
    logger.info("Mean head outputs collected.")

    general_set = bridge_heads["general"]
    rng = random.Random(args.seed)

    with HeadMaskManager(model, n_layers, n_heads, head_dim) as mask_mgr:
        for lang in args.langs:
            try:
                records = load_filtered_records(args.input_root, args.model_short, lang)
            except FileNotFoundError:
                logger.warning("Ablation: correct records not found for lang=%s -- skipping.", lang)
                continue
            records = sample_records(records, args.ablation_sample_size, args.seed)
            items = _collect_valid_items_ablation(model, tokenizer, records, args.gold_mode)
            logger.info("Ablation lang=%s: %d/%d valid items.", lang, len(items), len(records))
            if not items:
                continue

            base_nlls = [item["nll_base"] for item in items]
            base_nll_mean = float(np.mean(base_nlls))

            specific_set = bridge_heads["specific"].get(lang, [])
            conditions: list[tuple[str, list[tuple[int, int]]]] = [
                ("general", general_set),
                ("specific", specific_set),
            ]

            for head_set_type, head_set in conditions:
                if not head_set:
                    logger.warning(
                        "Ablation lang=%s type=%s: empty head set -- skipping.",
                        lang, head_set_type,
                    )
                    continue
                k = len(head_set)

                # Bridge head ablation
                bth_nlls, bth_per_item = _ablation_nlls(
                    model, tokenizer, items, head_set, mean_values, mask_mgr
                )
                delta_bth = [x["delta"] for x in bth_per_item]
                bth_stats = _summary_stats(delta_bth)
                bth_ci_lo, bth_ci_hi = bootstrap_ci(delta_bth, seed=args.seed)

                # Save per-item NLL for bridge head ablation
                _save_results(
                    bth_per_item,
                    out_dir / f"ablation_per_item_{lang}_{head_set_type}_bth",
                )

                results.append({
                    "model": args.model_short,
                    "lang": lang,
                    "head_set_type": head_set_type,
                    "baseline_type": "bth",
                    "intervention": "mean_ablation",
                    "k": k,
                    "sample_size": len(items),
                    "base_nll_mean": base_nll_mean,
                    "intervened_nll_mean": float(np.mean(bth_nlls)),
                    "delta_nll_mean": bth_stats["mean"],
                    "delta_nll_std": bth_stats["std"],
                    "delta_nll_ci_lo": bth_ci_lo,
                    "delta_nll_ci_hi": bth_ci_hi,
                })

                # Layer-matched random baseline (same k, --random-repeat repeats)
                rand_deltas: list[float] = []
                rand_per_item_all: list[dict[str, Any]] = []
                for repeat_i in range(args.random_repeat):
                    rand_set = sample_layer_matched_heads(head_set, n_heads, rng)
                    r_nlls, r_per_item = _ablation_nlls(
                        model, tokenizer, items, rand_set, mean_values, mask_mgr
                    )
                    rand_deltas.extend([x["delta"] for x in r_per_item])
                    rand_per_item_all.extend(
                        [{**x, "repeat": repeat_i} for x in r_per_item]
                    )
                rand_stats = _summary_stats(rand_deltas)
                rand_ci_lo, rand_ci_hi = bootstrap_ci(rand_deltas, seed=args.seed)

                # Save per-item NLL for random baseline
                _save_results(
                    rand_per_item_all,
                    out_dir / f"ablation_per_item_{lang}_{head_set_type}_random",
                )

                results.append({
                    "model": args.model_short,
                    "lang": lang,
                    "head_set_type": head_set_type,
                    "baseline_type": "layer_matched_random",
                    "intervention": "mean_ablation",
                    "k": k,
                    "sample_size": len(items),
                    "base_nll_mean": base_nll_mean,
                    "intervened_nll_mean": base_nll_mean + rand_stats["mean"],
                    "delta_nll_mean": rand_stats["mean"],
                    "delta_nll_std": rand_stats["std"],
                    "delta_nll_ci_lo": rand_ci_lo,
                    "delta_nll_ci_hi": rand_ci_hi,
                })

                logger.info(
                    "lang=%-4s  type=%-8s  k=%d  "
                    "BTH_delta=%.4f [%.4f, %.4f]  rand_delta=%.4f [%.4f, %.4f]",
                    lang, head_set_type, k,
                    bth_stats["mean"], bth_ci_lo, bth_ci_hi,
                    rand_stats["mean"], rand_ci_lo, rand_ci_hi,
                )

    out_dir.mkdir(parents=True, exist_ok=True)
    _save_results(results, out_dir / "ablation_results")
    return results


# ---------------------------------------------------------------------------
# Stage 3: Scaling Amplification
# ---------------------------------------------------------------------------

def run_scaling_amplification(
    lang: str,
    args: argparse.Namespace,
    model,
    tokenizer,
    bridge_heads: dict[str, Any],
    out_dir: Path,
    logger,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    specific_heads = bridge_heads["specific"].get(lang, [])
    if not specific_heads:
        logger.warning("No specific heads for lang=%s -- skipping scaling.", lang)
        return results

    n_layers = _get_num_layers(model)
    n_heads = _get_num_heads(model)
    head_dim = _get_head_dim(model)

    head_selection = getattr(args, "head_selection", "bridge")
    if head_selection == "random":
        rng = random.Random(f"{args.seed}-scaling-{lang}")
        amp_heads = sample_random_head_set(specific_heads, n_layers, n_heads, rng)
        logger.info(
            "Scaling lang=%s: using RANDOM head set (k=%d) instead of specific heads.",
            lang, len(amp_heads),
        )
    else:
        amp_heads = specific_heads

    try:
        records = load_incorrect_records(args.input_root, args.model_short, lang)
    except FileNotFoundError:
        logger.warning("Scaling: incorrect records not found for lang=%s -- skipping.", lang)
        return results
    records = _apply_scaling_sample(records, args.scaling_max_sample, args.seed)
    logger.info(
        "Scaling lang=%s: %d records  specific_heads=%d",
        lang, len(records), len(specific_heads),
    )
    if not records:
        return results

    items: list[dict] = []
    for record in records:
        labels = _gold_labels(record)
        if not labels:
            continue
        prompt = wrap_prompt(record["prompts"]["two_hop"], record["lang"])
        items.append({"id": record["id"], "prompt": prompt, "labels": labels})

    logger.info("Scaling lang=%s: %d valid items.", lang, len(items))
    if not items:
        return results

    with HeadMaskManager(model, n_layers, n_heads, head_dim) as mask_mgr:
        for alpha in args.alpha_list:
            correct = 0
            for item_i, item in enumerate(items, 1):
                mask_mgr.reset_masks()
                mask_mgr.apply_head_set_scaling(amp_heads, alpha)
                pred = predict_next_tokens(
                    model, tokenizer, item["prompt"], n_tokens=args.n_generate_tokens
                )
                mask_mgr.reset_masks()
                if check_answer(pred.split("\n")[0].strip(), item["labels"]):
                    correct += 1
                if item_i % 20 == 0 or item_i == len(items):
                    logger.info(
                        "  Scaling lang=%s alpha=%.2f  item %d/%d  acc_so_far=%.3f",
                        lang, alpha, item_i, len(items), correct / item_i,
                    )

            acc = correct / len(items)
            results.append({
                "model": args.model_short,
                "lang": lang,
                "head_set_type": "specific",
                "head_selection": head_selection,
                "intervention": "scaling",
                "alpha": alpha,
                "k": len(amp_heads),
                "n_items": len(items),
                "n_correct": correct,
                "amplified_accuracy": acc,
            })
            logger.info(
                "Scaling lang=%-4s  alpha=%.2f  head_selection=%s  amplified_acc=%.4f  (%d/%d)",
                lang, alpha, head_selection, acc, correct, len(items),
            )

    return results


# ---------------------------------------------------------------------------
# Stage 4: Cross-lingual Transfer Accuracy
# ---------------------------------------------------------------------------

def _load_cross_lingual_file(
    cross_lingual_root: Path,
    model_short: str,
    tgt_lang: str,
) -> list[dict]:
    fname = f"en_correct_{tgt_lang}_incorrect_{model_short}.json"
    path = cross_lingual_root / model_short / "filtered" / "cross_lang" / fname
    if not path.exists():
        raise FileNotFoundError(f"Cross-lingual file not found: {path}")
    return load_json(path)


def run_transfer_accuracy(
    tgt_lang: str,
    args: argparse.Namespace,
    model,
    tokenizer,
    bridge_heads: dict[str, Any],
    logger,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    general_heads = bridge_heads["general"]
    if not general_heads:
        logger.warning("No general heads -- skipping transfer for tgt_lang=%s.", tgt_lang)
        return results

    try:
        records = _load_cross_lingual_file(args.cross_lingual_root, args.model_short, tgt_lang)
    except FileNotFoundError as exc:
        logger.warning("Skipping transfer tgt_lang=%s: %s", tgt_lang, exc)
        return results

    records = _apply_scaling_sample(records, args.scaling_max_sample, args.seed)
    logger.info(
        "Transfer tgt_lang=%s: %d records  general_heads=%d",
        tgt_lang, len(records), len(general_heads),
    )
    if not records:
        return results

    n_layers = _get_num_layers(model)
    n_heads = _get_num_heads(model)
    head_dim = _get_head_dim(model)

    head_selection = getattr(args, "head_selection", "bridge")
    if head_selection == "random":
        rng = random.Random(f"{args.seed}-transfer-{tgt_lang}")
        amp_heads = sample_random_head_set(general_heads, n_layers, n_heads, rng)
        logger.info(
            "Transfer tgt_lang=%s: using RANDOM head set (k=%d) instead of general heads.",
            tgt_lang, len(amp_heads),
        )
    else:
        amp_heads = general_heads

    items: list[dict] = []
    for rec in records:
        lang_data = rec["langs"].get(tgt_lang)
        if lang_data is None:
            continue
        prompt = wrap_prompt(lang_data["prompts"]["two_hop"], tgt_lang)
        labels = [
            str(v).strip()
            for v in [lang_data.get("e3_label"), lang_data.get("e3_label_en")]
            if v and str(v).strip()
        ]
        if not labels:
            continue
        items.append({"id": rec["id"], "prompt": prompt, "labels": labels})

    logger.info("Transfer tgt_lang=%s: %d valid items.", tgt_lang, len(items))
    if not items:
        return results

    with HeadMaskManager(model, n_layers, n_heads, head_dim) as mask_mgr:
        for alpha in args.alpha_list:
            correct = 0
            for item_i, item in enumerate(items, 1):
                mask_mgr.reset_masks()
                mask_mgr.apply_head_set_scaling(amp_heads, alpha)
                pred = predict_next_tokens(
                    model, tokenizer, item["prompt"], n_tokens=args.n_generate_tokens
                )
                mask_mgr.reset_masks()
                if check_answer(pred.split("\n")[0].strip(), item["labels"]):
                    correct += 1
                if item_i % 20 == 0 or item_i == len(items):
                    logger.info(
                        "  Transfer tgt_lang=%s alpha=%.2f  item %d/%d  acc_so_far=%.3f",
                        tgt_lang, alpha, item_i, len(items), correct / item_i,
                    )

            acc = correct / len(items)
            results.append({
                "model": args.model_short,
                "tgt_lang": tgt_lang,
                "head_set_type": "general",
                "head_selection": head_selection,
                "intervention": "scaling",
                "alpha": alpha,
                "k": len(amp_heads),
                "n_items": len(items),
                "n_correct": correct,
                "base_accuracy": 0.0,
                "amplified_accuracy": acc,
                "acc_delta": acc,
            })
            logger.info(
                "Transfer tgt_lang=%-4s  alpha=%.2f  head_selection=%s  amplified_acc=%.4f  (%d/%d)",
                tgt_lang, alpha, head_selection, acc, correct, len(items),
            )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    logger = setup_logging(f"step6_Bridge_Head_validation_{args.model_short}", LOG_DIR)
    logger.info(
        "Step 6 -- model=%s  langs=%s  stage=%s  head_selection=%s",
        args.model_short, args.langs, args.stage, args.head_selection,
    )

    step5_dir = args.step5_root
    summary_dir = args.output_root / args.model_short
    summary_dir.mkdir(parents=True, exist_ok=True)

    bridge_heads = load_final_bridge_heads(step5_dir, args.model_short)
    logger.info(
        "Loaded bridge heads: general=%d  specific=%s",
        len(bridge_heads["general"]),
        {k: len(v) for k, v in bridge_heads["specific"].items()},
    )

    # Stage 1: Jaccard (no model needed)
    jaccard_results: list[dict] = []
    if args.stage in ("jaccard", "all"):
        logger.info("=== Stage 1: Jaccard Overlap Analysis ===")
        jaccard_results = run_jaccard_analysis(
            bridge_heads, langs=args.langs, out_dir=summary_dir, logger=logger,
        )

    if args.stage == "jaccard":
        logger.info("Stage 'jaccard' complete. Skipping model load.")
        return

    # Model load
    logger.info("Loading model: %s", args.model)
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        device="auto",
        torch_dtype=args.torch_dtype,
        hf_token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
        attn_implementation="eager",
    )

    all_ablation: list[dict] = []
    all_scaling: list[dict] = []
    all_transfer: list[dict] = []

    # Stage 3/4 outputs get a distinct filename suffix when amplifying a random
    # (non-bridge) head group, so they never overwrite the original bridge-head
    # results on disk.
    head_sel_suffix = "" if args.head_selection == "bridge" else "_random_heads"

    # Stage 2: Mean Ablation
    if args.stage in ("ablation", "all"):
        logger.info("=== Stage 2: Mean Ablation ===")
        try:
            all_ablation = run_mean_ablation(
                args, model, tokenizer, bridge_heads, out_dir=summary_dir, logger=logger,
            )
        except Exception as exc:
            logger.error("Mean ablation failed: %s", exc, exc_info=True)

    # Stage 3: Scaling Amplification
    if args.stage in ("scaling", "all"):
        for lang in args.langs:
            out_dir = args.output_root / args.model_short / lang
            out_dir.mkdir(parents=True, exist_ok=True)
            logger.info("=== Stage 3: Scaling Amplification -- lang=%s ===", lang)
            try:
                amp = run_scaling_amplification(
                    lang, args, model, tokenizer, bridge_heads, out_dir, logger
                )
                all_scaling.extend(amp)
                _save_results(amp, out_dir / f"scaling_results{head_sel_suffix}")
            except Exception as exc:
                logger.error("Scaling failed for lang=%s: %s", lang, exc, exc_info=True)

    # Stage 4: Cross-lingual Transfer Accuracy
    if args.stage in ("transfer", "all"):
        tgt_langs = [l for l in args.langs if l != "en"]
        if not tgt_langs:
            logger.warning("Stage 4 skipped: no non-EN languages in --langs.")
        for tgt_lang in tgt_langs:
            out_dir = args.output_root / args.model_short / tgt_lang
            out_dir.mkdir(parents=True, exist_ok=True)
            logger.info("=== Stage 4: Transfer Accuracy -- tgt_lang=%s ===", tgt_lang)
            try:
                tr = run_transfer_accuracy(
                    tgt_lang, args, model, tokenizer, bridge_heads, logger
                )
                all_transfer.extend(tr)
                _save_results(tr, out_dir / f"transfer_accuracy_en2{tgt_lang}{head_sel_suffix}")
            except Exception as exc:
                logger.error("Transfer failed for tgt_lang=%s: %s", tgt_lang, exc, exc_info=True)

    # Aggregate summaries
    if all_ablation:
        pd.DataFrame(all_ablation).to_csv(summary_dir / "ablation_all_langs.csv", index=False)
    if all_scaling:
        pd.DataFrame(all_scaling).to_csv(
            summary_dir / f"scaling_all_langs{head_sel_suffix}.csv", index=False
        )
    if all_transfer:
        pd.DataFrame(all_transfer).to_csv(
            summary_dir / f"transfer_accuracy_all_langs{head_sel_suffix}.csv", index=False
        )

    summary = {
        "model": args.model_short,
        "stage": args.stage,
        "langs": args.langs,
        "alpha_list": args.alpha_list,
        "ablation_sample_size": args.ablation_sample_size,
        "calibration_size": args.calibration_size,
        "scaling_max_sample": args.scaling_max_sample,
        "head_selection": args.head_selection,
        "n_general_heads": len(bridge_heads["general"]),
        "n_specific_heads": {k: len(v) for k, v in bridge_heads["specific"].items()},
        "n_jaccard_pairs": len(jaccard_results),
        "n_ablation_records": len(all_ablation),
        "n_scaling_records": len(all_scaling),
        "n_transfer_records": len(all_transfer),
    }
    save_json(summary, summary_dir / f"validation_summary{head_sel_suffix}.json")
    logger.info(
        "Step 6 complete.  jaccard=%d  ablation=%d  scaling=%d  transfer=%d",
        len(jaccard_results), len(all_ablation), len(all_scaling), len(all_transfer),
    )


if __name__ == "__main__":
    main()
