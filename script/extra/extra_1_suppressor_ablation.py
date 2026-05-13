"""Extra 1: Suppressor vs Supporter Group Ablation for LLaMA ko/zh specific heads.

Hypothesis (from Step 5/6 analysis):
    LLaMA ko/zh specific heads whose delta_Mono < 0 (ablating them *improves* EN
    two-hop NLL) may act as "suppressors" — inhibiting English reasoning pathways
    to route processing through a non-English pivot.

This experiment validates that hypothesis causally:

    1. Load Step 5 patchscopes-filtered specific heads for ko and zh (LLaMA only).
       Join delta_Mono values from Stage 2 ablation scores.
    2. Split heads by sign of delta_Mono:
         Suppressor group : delta_Mono < 0
         Supporter group  : delta_Mono >= 0
    3. Ablate each group *simultaneously* using the same Stage 2 mean-ablation
       protocol as step6_Bridge_Head_validation.py while measuring:
         - delta_EN   : change in EN two-hop NLL  (positive = degraded EN)
         - delta_LANG : change in KO/ZH two-hop NLL (positive = degraded KO/ZH)
       Calibration always uses en + ko/ja/zh/es; NLL gold is extracted from
       eval.two_hop_pred.
    4. Compare the two groups.

Expected result if hypothesis holds:
    Suppressor ablation → delta_EN < 0 (EN improves), delta_LANG > 0 (KO/ZH degrades)
    Supporter  ablation → delta_EN > 0,                delta_LANG > 0

Output
------
    output/extra/extra_1_suppressor_ablation/{model_short}/
        {lang}_group_ablation.json   # per-group delta summary
        {lang}_group_ablation.csv

Usage
-----
    python script/extra/extra_1_suppressor_ablation.py \\
        --model meta-llama/Llama-3.1-70B --model-short llama31_70 \\
        --langs ko zh

    # CPU-only re-analysis (reuse existing NLL; re-split threshold only)
    python script/extra/extra_1_suppressor_ablation.py \\
        --model meta-llama/Llama-3.1-70B --model-short llama31_70 \\
        --langs ko zh --reanalyse-only
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"

import sys
sys.path.insert(0, str(PROJECT_ROOT / "script"))

from utils.common import save_json, set_seed, setup_logging
from utils.model_utils import load_model_and_tokenizer
from utils.prompt_utils import wrap_prompt
from utils.bridge_utils import (
    extract_answer_span,
    load_filtered_records,
    sample_records,
)
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
        description="Extra 1: Suppressor vs Supporter group ablation for specific heads."
    )
    p.add_argument("--model-short", default="llama31_70",)
    p.add_argument("--model", default="meta-llama/Llama-3.1-70B",
                   help="HuggingFace model id or local path. Required unless --reanalyse-only.")
    p.add_argument("--langs", nargs="+", default=["ko", "zh"],
                   help="Languages to analyse (default: ko zh).")
    p.add_argument("--step5-root", type=Path,
                   default=PROJECT_ROOT / "output" / "step5_BridgeHead_Ablation_Patchscopes",
                   help="Root of Step 5 output.")
    p.add_argument("--output-root", type=Path,
                   default=PROJECT_ROOT / "output" / "extra" / "extra_1_suppressor_ablation")
    p.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--ablation-n", type=int, default=100,
                   help="Correct records per language for ablation test (default: 100).")
    p.add_argument("--calib-n", type=int, default=50,
                   help="Correct two-hop records per calibration language (default: 50).")
    p.add_argument("--gold-mode", default="first_span",
                   choices=["first_span", "first_word", "first_token"],
                   help="Gold extraction mode for ablation NLL from eval.two_hop_pred.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hf-token", default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--torch-dtype", default="auto")
    p.add_argument("--reanalyse-only", action="store_true",
                   help="Skip model inference; re-read existing raw NLL JSON and recompute summary.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _out_dir(args: argparse.Namespace) -> Path:
    d = args.output_root / args.model_short
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_final_heads_with_delta_mono(args: argparse.Namespace, lang: str) -> dict[str, dict]:
    """Load patchscopes-filtered specific heads and join delta_Mono from Step 5 ablation scores.

    Uses only heads that survived the full Step 5 pipeline:
      Stage 2 ablation (delta_XL > 0) → Stage 3 Patchscopes (successes > 0).

    Returns a dict "(layer,head)" → {layer, head, delta_XL, delta_Mono, ...}
    restricted to the patchscopes-filtered set.
    """
    # Stage 3: patchscopes-filtered heads (final set)
    ps_path = (
        args.step5_root / args.model_short
        / "stage3_patchscopes" / lang / "filtered_heads.json"
    )
    if not ps_path.exists():
        raise FileNotFoundError(
            f"Patchscopes filtered heads not found for {lang}: {ps_path}\n"
            "Run step5 --stage patchscopes first."
        )
    ps_data = json.loads(ps_path.read_text())
    final_pairs: set[tuple[int, int]] = {
        (int(h["layer"]), int(h["head"])) for h in ps_data["heads"]
    }
    if not final_pairs:
        raise ValueError(f"[{lang}] No patchscopes-filtered heads found.")

    # Stage 2: ablation scores (for delta_Mono values)
    abl_path = (
        args.step5_root / args.model_short
        / "stage2_ablation" / lang / "ablation_scores.json"
    )
    if not abl_path.exists():
        raise FileNotFoundError(
            f"Ablation scores not found for {lang}: {abl_path}\n"
            "Run step5 --stage ablation first."
        )
    abl_data = json.loads(abl_path.read_text())

    # Keep only heads present in both stages
    result: dict[str, dict] = {
        key: v
        for key, v in abl_data["heads"].items()
        if (int(v["layer"]), int(v["head"])) in final_pairs
    }
    if not result:
        raise ValueError(
            f"[{lang}] No overlap between patchscopes-filtered heads and ablation scores. "
            "Verify that both Step 5 stages were run for the same model."
        )
    return result


def _split_groups(
    head_scores: dict[str, dict],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Split heads into suppressor (delta_Mono < 0) and supporter (delta_Mono >= 0)."""
    suppressors: list[tuple[int, int]] = []
    supporters:  list[tuple[int, int]] = []
    for v in head_scores.values():
        pair = (int(v["layer"]), int(v["head"]))
        if v["delta_Mono"] < 0:
            suppressors.append(pair)
        else:
            supporters.append(pair)
    return suppressors, supporters


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


# ---------------------------------------------------------------------------
# Calibration mean (same procedure as Step 6 Stage 2)
# ---------------------------------------------------------------------------

def _build_calib_means_step6_style(
    args: argparse.Namespace,
    model,
    tokenizer,
    logger,
) -> dict[tuple[int, int], float]:
    """Build calibration means exactly like Step 6 Stage 2.

    Step 6 uses correct two-hop prompts sampled from EN plus ko/ja/zh/es. It
    does not reuse Step 5 calibration means because those are built from
    TH/FH/SH over all languages and therefore represent a different calibration
    distribution.
    """
    n_layers = _get_num_layers(model)
    n_heads = _get_num_heads(model)
    head_dim = _get_head_dim(model)

    calib_langs = ["en", "ko", "ja", "zh", "es"]
    prompts: list[str] = []
    for clang in calib_langs:
        try:
            records = load_filtered_records(args.input_root, args.model_short, clang)
        except FileNotFoundError:
            logger.warning("Calibration: correct records not found for lang=%s -- skipping.", clang)
            continue
        sample = sample_records(records, args.calib_n, args.seed)
        for rec in sample:
            prompts.append(wrap_prompt(rec["prompts"]["two_hop"], rec["lang"]))

    logger.info(
        "Calibration: %d prompts from %d langs (target %d x %d = %d).",
        len(prompts), len(calib_langs),
        len(calib_langs), args.calib_n, len(calib_langs) * args.calib_n,
    )

    mean_values = collect_mean_head_outputs(
        model, tokenizer, prompts, n_layers, n_heads, head_dim
    )
    logger.info("Mean head outputs collected.")
    return mean_values


# ---------------------------------------------------------------------------
# Step 6-style ablation item loading
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


# ---------------------------------------------------------------------------
# Group NLL measurement
# ---------------------------------------------------------------------------

@torch.no_grad()
def _measure_group_nll(
    model,
    tokenizer,
    items: list[dict[str, Any]],
    heads: list[tuple[int, int]],
    mean_vals: dict[tuple[int, int], float],
    mask_mgr: HeadMaskManager,
) -> tuple[float, float, list[dict[str, Any]]]:
    """Measure mean NLL (vanilla) and mean NLL (group-ablated) over records.

    All heads in `heads` are ablated simultaneously.

    Returns
    -------
    (nll_vanilla_mean, nll_ablated_mean, per_item)
        per_item : list of {nll_vanilla, nll_ablated, delta} per valid record
    """
    nll_ablated_list: list[float] = []
    per_item: list[dict[str, Any]] = []

    for item in items:
        mask_mgr.reset_masks()
        mask_mgr.apply_head_set_ablation(heads, mean_vals)
        nll_a = compute_nll_eval(model, tokenizer, item["prompt"], item["gold"])
        if math.isnan(nll_a):
            continue

        nll_ablated_list.append(nll_a)
        per_item.append({
            "id": item["id"],
            "nll_vanilla": item["nll_base"],
            "nll_ablated": nll_a,
            "delta": nll_a - item["nll_base"],
        })

    mask_mgr.reset_masks()
    if not per_item:
        return float("nan"), float("nan"), []
    return statistics.mean([x["nll_vanilla"] for x in per_item]), statistics.mean(nll_ablated_list), per_item


# ---------------------------------------------------------------------------
# Main per-language experiment
# ---------------------------------------------------------------------------

def run_lang(
    args: argparse.Namespace,
    model,
    tokenizer,
    lang: str,
    mean_vals: dict[tuple[int, int], float],
    logger,
) -> dict[str, Any]:
    """Run suppressor/supporter group ablation for one language."""
    logger.info("=== [%s] Loading patchscopes-filtered specific heads (Step 5 Stage 3) ===", lang)
    head_scores = _load_final_heads_with_delta_mono(args, lang)
    suppressors, supporters = _split_groups(head_scores)

    logger.info(
        "[%s] Suppressor heads (delta_Mono<0): %d | Supporter heads (delta_Mono>=0): %d",
        lang, len(suppressors), len(supporters),
    )
    for label, group in [("suppressor", suppressors), ("supporter", supporters)]:
        logger.info("  %s: %s", label, sorted(group))

    # Load Step 6-style ablation records: sampled independently from each
    # language's correct two-hop file, then filtered by valid gold/NLL.
    en_records = sample_records(
        load_filtered_records(args.input_root, args.model_short, "en"),
        args.ablation_n,
        args.seed,
    )
    lang_records = sample_records(
        load_filtered_records(args.input_root, args.model_short, lang),
        args.ablation_n,
        args.seed,
    )
    en_items = _collect_valid_items_ablation(model, tokenizer, en_records, args.gold_mode)
    lang_items = _collect_valid_items_ablation(model, tokenizer, lang_records, args.gold_mode)
    logger.info(
        "[%s] Ablation items: EN %d/%d valid | %s %d/%d valid",
        lang, len(en_items), len(en_records), lang.upper(), len(lang_items), len(lang_records),
    )

    n_layers = _get_num_layers(model)
    n_heads  = _get_num_heads(model)
    head_dim = _get_head_dim(model)

    results: dict[str, Any] = {
        "lang": lang,
        "model": args.model_short,
        "n_suppressor": len(suppressors),
        "n_supporter":  len(supporters),
        "suppressor_heads": [list(h) for h in suppressors],
        "supporter_heads":  [list(h) for h in supporters],
        "n_records_en": len(en_items),
        f"n_records_{lang}": len(lang_items),
        "groups": {},
    }

    with HeadMaskManager(model, n_layers, n_heads, head_dim) as mask_mgr:
        for group_label, group_heads in [
            ("suppressor", suppressors),
            ("supporter",  supporters),
        ]:
            if not group_heads:
                logger.info("[%s] %s group is empty — skipping.", lang, group_label)
                results["groups"][group_label] = {"skipped": True, "reason": "empty group"}
                continue

            logger.info("[%s] %s group ablation (%d heads) — EN measurement …",
                        lang, group_label, len(group_heads))

            # EN two-hop NLL
            nll_en_v, nll_en_a, per_item_en = _measure_group_nll(
                model, tokenizer,
                en_items, group_heads, mean_vals, mask_mgr,
            )
            delta_en = (nll_en_a - nll_en_v) if not math.isnan(nll_en_v) else float("nan")
            delta_en_list = [x["delta"] for x in per_item_en]
            ci_en_lo, ci_en_hi = bootstrap_ci(delta_en_list, seed=args.seed)

            logger.info("[%s] %s group ablation — %s measurement …",
                        lang, group_label, lang.upper())

            # LANG two-hop NLL
            nll_lang_v, nll_lang_a, per_item_lang = _measure_group_nll(
                model, tokenizer,
                lang_items, group_heads, mean_vals, mask_mgr,
            )
            delta_lang = (nll_lang_a - nll_lang_v) if not math.isnan(nll_lang_v) else float("nan")
            delta_lang_list = [x["delta"] for x in per_item_lang]
            ci_lang_lo, ci_lang_hi = bootstrap_ci(delta_lang_list, seed=args.seed)

            logger.info(
                "[%s] %s | delta_EN=%.4f [%.4f, %.4f]  delta_%s=%.4f [%.4f, %.4f]",
                lang, group_label,
                delta_en, ci_en_lo, ci_en_hi,
                lang.upper(), delta_lang, ci_lang_lo, ci_lang_hi,
            )

            results["groups"][group_label] = {
                "n_heads":      len(group_heads),
                "n_records_en": len(per_item_en),
                f"n_records_{lang}": len(per_item_lang),
                "nll_en_vanilla":  nll_en_v,
                "nll_en_ablated":  nll_en_a,
                "delta_EN":        delta_en,
                "delta_EN_ci_lo":  ci_en_lo,
                "delta_EN_ci_hi":  ci_en_hi,
                "per_item_en":     per_item_en,
                f"nll_{lang}_vanilla": nll_lang_v,
                f"nll_{lang}_ablated": nll_lang_a,
                f"delta_{lang.upper()}": delta_lang,
                f"delta_{lang.upper()}_ci_lo": ci_lang_lo,
                f"delta_{lang.upper()}_ci_hi": ci_lang_hi,
                f"per_item_{lang}": per_item_lang,
            }

    return results


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_lang_results(results: dict[str, Any], out_dir: Path, lang: str) -> None:
    # Extract per-item lists and save as JSONL; strip from main JSON to keep it compact
    results_clean: dict[str, Any] = {k: v for k, v in results.items() if k != "groups"}
    results_clean["groups"] = {}
    for group_label, g in results["groups"].items():
        if g.get("skipped"):
            results_clean["groups"][group_label] = g
            continue
        g_clean = {k: v for k, v in g.items() if not k.startswith("per_item_")}
        results_clean["groups"][group_label] = g_clean

        for pi_key in ("per_item_en", f"per_item_{lang}"):
            pi_records = g.get(pi_key, [])
            if pi_records:
                pi_path = out_dir / f"{lang}_{group_label}_{pi_key}.jsonl"
                with pi_path.open("w", encoding="utf-8") as f:
                    for item in pi_records:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                print(f"  Saved per-item: {pi_path}")

    json_path = out_dir / f"{lang}_group_ablation.json"
    save_json(results_clean, json_path)

    # Flat CSV with CI columns
    rows = []
    for group_label, g in results["groups"].items():
        if g.get("skipped"):
            continue
        row = {
            "lang": lang,
            "model": results["model"],
            "group": group_label,
            "n_heads": g["n_heads"],
            "n_records_en": g.get("n_records_en"),
            f"n_records_{lang}": g.get(f"n_records_{lang}"),
            "delta_EN": g.get("delta_EN"),
            "delta_EN_ci_lo": g.get("delta_EN_ci_lo"),
            "delta_EN_ci_hi": g.get("delta_EN_ci_hi"),
            f"delta_{lang.upper()}": g.get(f"delta_{lang.upper()}"),
            f"delta_{lang.upper()}_ci_lo": g.get(f"delta_{lang.upper()}_ci_lo"),
            f"delta_{lang.upper()}_ci_hi": g.get(f"delta_{lang.upper()}_ci_hi"),
            "nll_en_vanilla": g.get("nll_en_vanilla"),
            "nll_en_ablated": g.get("nll_en_ablated"),
            f"nll_{lang}_vanilla": g.get(f"nll_{lang}_vanilla"),
            f"nll_{lang}_ablated": g.get(f"nll_{lang}_ablated"),
        }
        rows.append(row)
    if rows:
        csv_path = out_dir / f"{lang}_group_ablation.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")


def _print_summary(all_results: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 62)
    print("  Suppressor vs Supporter Group Ablation — Summary")
    print("=" * 62)
    for res in all_results:
        lang = res["lang"]
        print(f"\n  Language: {lang.upper()}")
        print(f"  Suppressor heads: {res['n_suppressor']}  |  Supporter heads: {res['n_supporter']}")
        print(f"  {'Group':<12}  {'delta_EN (95% CI)':>26}  {'delta_'+lang.upper()+' (95% CI)':>28}  Interpretation")
        print("  " + "-" * 82)
        for group_label, g in res["groups"].items():
            if g.get("skipped"):
                print(f"  {group_label:<12}  (empty)")
                continue
            d_en      = g.get("delta_EN", float("nan"))
            ci_en_lo  = g.get("delta_EN_ci_lo", float("nan"))
            ci_en_hi  = g.get("delta_EN_ci_hi", float("nan"))
            d_lang    = g.get(f"delta_{lang.upper()}", float("nan"))
            ci_lang_lo = g.get(f"delta_{lang.upper()}_ci_lo", float("nan"))
            ci_lang_hi = g.get(f"delta_{lang.upper()}_ci_hi", float("nan"))
            en_str   = f"{d_en:+.4f} [{ci_en_lo:+.4f}, {ci_en_hi:+.4f}]"
            lang_str = f"{d_lang:+.4f} [{ci_lang_lo:+.4f}, {ci_lang_hi:+.4f}]"
            if group_label == "suppressor":
                interp = "✓ hypothesis" if (d_en < 0 and d_lang > 0) else "✗ not supported"
            else:
                interp = "expected: both +" if d_en > 0 else ""
            print(f"  {group_label:<12}  {en_str:>26}  {lang_str:>28}  {interp}")
    print("=" * 62 + "\n")


# ---------------------------------------------------------------------------
# Re-analyse only (no model, read existing JSON)
# ---------------------------------------------------------------------------

def reanalyse(args: argparse.Namespace, logger) -> None:
    """Re-read existing raw JSON outputs and print summary."""
    all_results = []
    for lang in args.langs:
        json_path = _out_dir(args) / f"{lang}_group_ablation.json"
        if not json_path.exists():
            logger.warning("No existing results for %s at %s", lang, json_path)
            continue
        results = json.loads(json_path.read_text())
        all_results.append(results)
        _save_lang_results(results, _out_dir(args), lang)
    if all_results:
        _print_summary(all_results)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    logger = setup_logging(f"extra1_suppressor_{args.model_short}", LOG_DIR)

    if args.reanalyse_only:
        logger.info("--reanalyse-only: skipping model load.")
        reanalyse(args, logger)
        return

    if not args.model:
        raise ValueError("--model is required unless --reanalyse-only is set.")

    logger.info("Loading model: %s", args.model)
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        device="auto",
        torch_dtype=args.torch_dtype,
        hf_token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
        attn_implementation="eager",
    )

    out_dir = _out_dir(args)
    mean_vals = _build_calib_means_step6_style(args, model, tokenizer, logger)
    all_results = []

    for lang in args.langs:
        try:
            results = run_lang(args, model, tokenizer, lang, mean_vals, logger)
        except FileNotFoundError as e:
            logger.error("%s", e)
            continue
        _save_lang_results(results, out_dir, lang)
        all_results.append(results)
        logger.info("[%s] Done. Results saved to %s", lang, out_dir)

    if all_results:
        _print_summary(all_results)

    logger.info("Extra 1 complete.")


if __name__ == "__main__":
    main()
