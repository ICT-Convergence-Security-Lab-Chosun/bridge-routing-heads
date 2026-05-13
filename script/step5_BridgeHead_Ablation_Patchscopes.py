"""Step 5: Bridge Head Ablation Filtering and Patchscopes Verification.

Two-stage pipeline to refine the Bridge Head candidates produced by Step 4
(BHS-based 1st-stage filtering):

  Stage 2 — Mean Ablation Filtering
    For each candidate head, replace its output with the calibration mean and
    measure delta-NLL across four conditions:
      XL-2H   : Cross-lingual two-hop (target language)
      Mono-2H : English two-hop (reference)
      FH_L    : First-hop in target language
      SH_L    : Second-hop in target language

    Pass criterion (1st-pass filter — intentionally permissive):
      delta_XL > 0   (ablation must degrade cross-lingual 2-hop; negative delta means
                      the head was irrelevant or suppressive, so discard it)

    FH, Mono, SH are measured and stored for analysis but not used as pass criteria.

    Heads passing are ranked by delta_XL descending.

    Language-General candidates additionally require:
      min(delta_XL across all langs) > 0   (must degrade XL 2-hop in EVERY language)

  Stage 2B — Question-Scoped Attention Diagnostics
    For Stage-2-passing heads, inspect the final input token's attention row,
    but only over tokens inside the raw Question text (between Question: and
    Answer:), excluding edge question marks.  Save per-language attention
    weights and the top two peaks.

  Stage 3 — Patchscopes Verification
    For each Stage-2-passing head, extract the full residual stream at the
    r1_noun token in the two-hop prompt.  Inject that hidden state into a
    "decode probe" prompt at the 2/3-depth layer and store whether the model
    generates the bridge entity (e2), along with the generated text.

    Heads pass Stage 3 if at least one Patchscopes probe generates e2.  No
    random-head baseline or baseline+k*std threshold is used.

Output layout
-------------
  output/step5_BridgeHead_Ablation_Patchscopes/{model_short}/
      calibration_means.json          # (layer, head) → mean scalar
      stage2_ablation/
          {lang}/ablation_scores.json      # per-head BSS + deltas
          {lang}/ablation_summary.csv
          {lang}/filtered_heads.json       # heads passing threshold
          general/ablation_scores.json
          general/filtered_heads.json
      stage2.5_attention/
          {lang}/question_attn_weights.json
      stage3_patchscopes/
          {lang}/patchscopes_scores.json   # per-head encoding_rate + generations
          {lang}/filtered_heads.json        # heads that generated e2 via Patchscopes
      final_bridge_heads.json              # complete pipeline result

Usage examples
--------------
# Step A — build calibration means (CPU-light GPU pass):
python script/step5_BridgeHead_Ablation_Patchscopes.py \\
    --model-short llama31_70 --model meta-llama/Llama-3.1-70B \\
    --langs ko zh ja es --stage calibrate

# Step B — mean ablation filtering:
python script/step5_BridgeHead_Ablation_Patchscopes.py \\
    --model-short llama31_70 --model meta-llama/Llama-3.1-70B \\
    --langs ko zh ja es --stage ablation

# Step C — question-scoped attention diagnostics:
python script/step5_BridgeHead_Ablation_Patchscopes.py \\
    --model-short llama31_70 --model meta-llama/Llama-3.1-70B \\
    --langs ko zh ja es --stage attention

# Step D — patchscopes verification:
python script/step5_BridgeHead_Ablation_Patchscopes.py \\
    --model-short llama31_70 --model meta-llama/Llama-3.1-70B \\
    --langs ko zh ja es --stage patchscopes

# Run all stages sequentially:
python script/step5_BridgeHead_Ablation_Patchscopes.py \\
    --model-short llama31_70 --model meta-llama/Llama-3.1-70B \\
    --langs ko zh ja es --stage all

# Re-apply ablation and Patchscopes filters without re-running the model:
python script/step5_BridgeHead_Ablation_Patchscopes.py \\
    --model-short llama31_70 --model meta-llama/Llama-3.1-70B \\
    --langs ko zh ja es --stage threshold
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"

import sys
sys.path.insert(0, str(PROJECT_ROOT / "script"))

from utils.common import load_json, save_json, set_seed, setup_logging
from utils.model_utils import load_model_and_tokenizer
from utils.prompt_utils import wrap_prompt
from utils.bridge_utils import (
    extract_answer_span,
    load_filtered_records,
    load_cross_lang_records,
    flatten_cross_lang_record,
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
# Constants
# ---------------------------------------------------------------------------

ALL_LANGS = ["en", "ko", "zh", "ja", "es"]
XL_LANGS  = ["ko", "zh", "ja", "es"]          # non-English (XL-2H targets)

# Patchscopes decode probe.  "x" is replaced with the injected hidden state.
_PATCHSCOPES_PROMPT = (
    "Syria: Syria is a country in the Middle East, "
    "Leonardo DiCaprio: Leonardo DiCaprio is an American actor, "
    "Samsung: Samsung is a South Korean multinational corporation, "
    "x:"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 5: Bridge Head Ablation Filtering + Patchscopes Diagnostics."
    )
    p.add_argument("--model-short", required=True)
    p.add_argument("--model", default=None,
                   help="HuggingFace model id or local path.  Required for all GPU stages.")
    p.add_argument("--langs", nargs="+", default=XL_LANGS,
                   help="Target XL languages (non-English).")
    p.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "data",
                   help="Root of filtered data (step3 output).")
    p.add_argument("--templates", type=Path,
                   default=PROJECT_ROOT / "script" / "config" / "relation_templates.json",
                   help="Relation templates used to locate the r1_noun token.")
    p.add_argument("--step4-root", type=Path,
                   default=PROJECT_ROOT / "output" / "step4_filtering_Bridge_head_Score",
                   help="Root of step4 head_sets output.")
    p.add_argument("--step4-formula", default="fh+th-sh",
                   choices=["fh+th-sh", "th-fh-sh"],
                   help="Which step4 formula's head_sets.jsonl to use as candidates.")
    p.add_argument("--output-root", type=Path,
                   default=PROJECT_ROOT / "output" / "step5_BridgeHead_Ablation_Patchscopes",
                   help="Root for all Step 5 outputs.")
    # Calibration
    p.add_argument("--calib-per-cond", type=int, default=30,
                   help="Calibration samples per (lang × condition) cell.")
    # Ablation
    p.add_argument("--ablation-n", type=int, default=50,
                   help="Correct TH samples per language for ablation measurement.")
    # Patchscopes
    p.add_argument("--patchscopes-n", type=int, default=50,
                   help="Correct TH samples per language for patchscopes.")
    p.add_argument("--patchscopes-gen-len", type=int, default=20,
                   help="Max new tokens generated in each patchscopes probe.")
    p.add_argument("--patchscopes-n-runs", type=int, default=3,
                   help="Number of independent generation runs per patchscopes probe.")
    # Stage selector
    p.add_argument("--stage", default="all",
                   choices=["calibrate", "ablation", "patchscopes", "attention", "threshold", "all"],
                   help="Which stage(s) to run. threshold re-applies saved filters only.")
    # Model loading
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hf-token", default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--torch-dtype", default="auto")
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _out_dir(args: argparse.Namespace) -> Path:
    d = args.output_root / args.model_short
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_step4_candidates(args: argparse.Namespace, logger) -> dict[str, Any]:
    """Load head_sets.jsonl from step4 and return the head-set dict.

    Returns
    -------
    dict mapping set_name → list of head records (layer, head, head_id, …)
    """
    jsonl_path = (
        args.step4_root / args.model_short / args.step4_formula / "head_sets.jsonl"
    )
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Step4 head_sets.jsonl not found: {jsonl_path}")

    head_sets: dict[str, list[dict]] = {}
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            head_sets[obj["set_name"]] = obj["heads"]
    logger.info("Loaded %d head sets from %s", len(head_sets), jsonl_path)
    return head_sets


def _records_to_tuples(records: list[dict]) -> list[tuple[int, int]]:
    return [(r["layer"], r["head"]) for r in records]


def _load_filtered_by_lang(args: argparse.Namespace, lang: str) -> list[dict]:
    return load_filtered_records(args.input_root, args.model_short, lang)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def build_calibration_prompts(args: argparse.Namespace, logger) -> list[str]:
    """Sample prompts from TH/FH/SH × all languages for mean calibration."""
    cond_keys = {
        "TH": "two_hop",
        "FH": "first_hop",
        "SH": "second_hop",
    }
    prompts: list[str] = []
    rng = random.Random(args.seed)

    for lang in ALL_LANGS:
        records = _load_filtered_by_lang(args, lang)
        rng.shuffle(records)
        pool = records[: args.calib_per_cond * len(cond_keys) * 2]  # headroom

        for cond, pkey in cond_keys.items():
            sampled = [r for r in pool if pkey in r.get("prompts", {})]
            sampled = sampled[: args.calib_per_cond]
            for rec in sampled:
                prompts.append(wrap_prompt(rec["prompts"][pkey], lang))

    logger.info("Built %d calibration prompts", len(prompts))
    return prompts


def run_calibrate(args: argparse.Namespace, model, tokenizer, logger) -> None:
    out = _out_dir(args)
    calib_path = out / "calibration_means.json"
    if calib_path.exists():
        logger.info("Calibration already done: %s", calib_path)
        return

    n_layers = _get_num_layers(model)
    n_heads  = _get_num_heads(model)
    head_dim = _get_head_dim(model)

    prompts = build_calibration_prompts(args, logger)
    logger.info("Computing mean head outputs on %d prompts …", len(prompts))
    mean_vals = collect_mean_head_outputs(model, tokenizer, prompts, n_layers, n_heads, head_dim)

    # Convert tuple keys to strings for JSON serialisation
    serialisable = {f"{l},{h}": v for (l, h), v in mean_vals.items()}
    save_json(serialisable, calib_path)
    logger.info("Saved calibration means: %s", calib_path)


def load_calibration_means(args: argparse.Namespace) -> dict[tuple[int, int], float]:
    calib_path = _out_dir(args) / "calibration_means.json"
    if not calib_path.exists():
        raise FileNotFoundError(
            "Calibration means not found.  Run --stage calibrate first."
        )
    raw = json.loads(calib_path.read_text())
    return {(int(k.split(",")[0]), int(k.split(",")[1])): float(v) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Ablation helpers
# ---------------------------------------------------------------------------

def _sample_correct(records: list[dict], n: int, seed: int,
                    cond_correct_key: str = "two_hop_correct") -> list[dict]:
    """Return up to n records where the target condition is correct."""
    correct = [r for r in records if r.get("eval", {}).get(cond_correct_key)]
    rng = random.Random(seed)
    rng.shuffle(correct)
    return correct[:n]


def _sample_correct_both(records: list[dict], n: int, seed: int) -> list[dict]:
    """Return records where two_hop AND first_hop are both correct."""
    correct = [
        r for r in records
        if r.get("eval", {}).get("two_hop_correct")
        and r.get("eval", {}).get("first_hop_correct")
    ]
    rng = random.Random(seed)
    rng.shuffle(correct)
    return correct[:n]


def _load_cross_lang_pairs(
    args: argparse.Namespace,
    lang: str,
    n: int,
) -> tuple[list[dict], list[dict]]:
    """Load pre-built en_{lang}_both_correct paired file and return
    (en_flat_records, lang_flat_records) of length up to n.

    Each flat record has the same schema as a normal per-lang filtered record
    (prompts, eval, e2_label_en, e2, …) so it is compatible with
    _measure_delta_nll without modification.
    """
    cross_recs = load_cross_lang_records(args.input_root, args.model_short, "en", lang)
    rng = random.Random(args.seed)
    rng.shuffle(cross_recs)
    selected = cross_recs[:n]
    en_flat   = [flatten_cross_lang_record(r, "en")  for r in selected]
    lang_flat = [flatten_cross_lang_record(r, lang)  for r in selected]
    return en_flat, lang_flat


@torch.no_grad()
def _measure_delta_nll(
    model,
    tokenizer,
    records: list[dict],
    prompt_key: str,
    pred_key: str,
    lang: str,
    layer: int,
    head: int,
    mean_val: float,
    mask_mgr: HeadMaskManager,
    gold_mode: str = "first_span",
) -> float:
    """Mean delta-NLL when ablating (layer, head) on the given records."""
    deltas: list[float] = []
    for rec in records:
        prompt = wrap_prompt(rec["prompts"][prompt_key], lang)
        raw_pred = rec.get("eval", {}).get(pred_key, "")
        gold = extract_answer_span(raw_pred, mode=gold_mode)
        if not gold:
            continue

        # Vanilla NLL
        mask_mgr.reset_masks()
        nll_vanilla = compute_nll_eval(model, tokenizer, prompt, gold)
        if math.isnan(nll_vanilla):
            continue

        # Ablated NLL
        mask_mgr.reset_masks()
        mask_mgr.set_ablation(layer, head, mean_val)
        nll_ablated = compute_nll_eval(model, tokenizer, prompt, gold)
        if math.isnan(nll_ablated):
            continue

        deltas.append(nll_ablated - nll_vanilla)

    return statistics.mean(deltas) if deltas else 0.0


# ---------------------------------------------------------------------------
# Stage 2A: Language-Specific ablation filtering
# ---------------------------------------------------------------------------

def run_ablation_specific(
    args: argparse.Namespace,
    model,
    tokenizer,
    head_sets: dict[str, Any],
    mean_vals: dict[tuple[int, int], float],
    logger,
) -> None:
    n_layers = _get_num_layers(model)
    n_heads  = _get_num_heads(model)
    head_dim = _get_head_dim(model)

    # Pre-load English records once — used only for SH (single-lang, no pairing needed)
    with HeadMaskManager(model, n_layers, n_heads, head_dim) as mask_mgr:
        for lang in args.langs:
            out_lang_dir = _out_dir(args) / "stage2_ablation" / lang
            out_lang_dir.mkdir(parents=True, exist_ok=True)
            scores_path   = out_lang_dir / "ablation_scores.json"

            set_name = f"H_Bridge_Specific_{lang.upper()}"
            candidates = _records_to_tuples(head_sets.get(set_name, []))
            if not candidates:
                logger.warning("No candidates for %s — skipping.", set_name)
                continue

            if scores_path.exists():
                logger.info("[%s] Ablation scores already exist — skipping.", lang)
                continue

            logger.info("[%s] Ablation: %d candidate heads", lang, len(candidates))

            # XL / Mono: load pre-built paired file (same hop_id, two languages)
            mono_flat, xl_flat = _load_cross_lang_pairs(args, lang, args.ablation_n)

            lang_records = _load_filtered_by_lang(args, lang)
            fh_samples  = _sample_correct(lang_records, args.ablation_n, args.seed,
                                          "first_hop_correct")
            sh_samples  = _sample_correct(lang_records, args.ablation_n, args.seed,
                                          "second_hop_correct")
            logger.info("[%s] paired XL/Mono=%d  FH=%d  SH=%d",
                        lang, len(xl_flat), len(fh_samples), len(sh_samples))

            scores: dict[str, dict] = {}
            pbar_cand = tqdm(candidates, desc=f"[{lang}] ablation", unit="head")
            for layer, head in pbar_cand:
                mean_val = mean_vals.get((layer, head), 0.0)
                key = f"({layer},{head})"

                delta_xl   = _measure_delta_nll(model, tokenizer, xl_flat,
                                                "two_hop",   "two_hop_pred",   lang,
                                                layer, head, mean_val, mask_mgr)
                delta_mono = _measure_delta_nll(model, tokenizer, mono_flat,
                                                "two_hop",   "two_hop_pred",   "en",
                                                layer, head, mean_val, mask_mgr)
                delta_fh   = _measure_delta_nll(model, tokenizer, fh_samples,
                                                "first_hop", "first_hop_pred", lang,
                                                layer, head, mean_val, mask_mgr)
                delta_sh   = _measure_delta_nll(model, tokenizer, sh_samples,
                                                "second_hop","second_hop_pred",lang,
                                                layer, head, mean_val, mask_mgr)
                scores[key] = {
                    "layer": layer, "head": head,
                    "delta_XL":   delta_xl,
                    "delta_Mono": delta_mono,
                    "delta_FH":   delta_fh,
                    "delta_SH":   delta_sh,
                }
                pbar_cand.set_postfix(dXL=f"{delta_xl:.4f}", dMono=f"{delta_mono:.4f}",
                                      dSH=f"{delta_sh:.4f}")
                logger.debug("[%s] %s  dXL=%.4f  dMono=%.4f  dSH=%.4f",
                             lang, key, delta_xl, delta_mono, delta_sh)

            # Save all scores (allows re-thresholding without GPU re-run)
            result = {
                "lang": lang,
                "model": args.model_short,
                "n_candidates": len(candidates),
                "heads": scores,
            }
            save_json(result, scores_path)
            logger.info("[%s] Saved ablation scores → %s", lang, scores_path)

    # Apply thresholds (separate so it can be re-run alone)
    _apply_ablation_threshold_specific(args, logger)


# ---------------------------------------------------------------------------
# Stage 2B: Language-General ablation filtering
# ---------------------------------------------------------------------------

def run_ablation_general(
    args: argparse.Namespace,
    model,
    tokenizer,
    head_sets: dict[str, Any],
    mean_vals: dict[tuple[int, int], float],
    logger,
) -> None:
    n_layers = _get_num_layers(model)
    n_heads  = _get_num_heads(model)
    head_dim = _get_head_dim(model)

    candidates = _records_to_tuples(head_sets.get("H_Bridge_General", []))
    if not candidates:
        logger.warning("No H_Bridge_General candidates — skipping general ablation.")
        return

    out_dir = _out_dir(args) / "stage2_ablation" / "general"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "ablation_scores.json"

    if scores_path.exists():
        logger.info("[General] Ablation scores already exist — skipping to threshold.")
        _apply_ablation_threshold_general(args, logger)
        return

    logger.info("[General] Ablation: %d candidate heads across %d languages",
                len(candidates), len(args.langs))

    # Per-language paired (en, lang) samples + SH samples
    paired_by_lang:    dict[str, tuple[list[dict], list[dict]]] = {}
    sh_samples_by_lang: dict[str, list[dict]] = {}
    for lang in args.langs:
        mono_flat, xl_flat = _load_cross_lang_pairs(args, lang, args.ablation_n)
        paired_by_lang[lang] = (mono_flat, xl_flat)
        records = _load_filtered_by_lang(args, lang)
        sh_samples_by_lang[lang] = _sample_correct(records, args.ablation_n, args.seed,
                                                   "second_hop_correct")
        logger.info("[General] [%s] %d paired XL/Mono  %d SH",
                    lang, len(xl_flat), len(sh_samples_by_lang[lang]))

    scores: dict[str, dict] = {}

    def _deltas_per_lang(layer, head, mean_val, mask_mgr_):
        """Per-lang deltas using hop_id-matched XL/Mono pairs.

        Returns (lang_xl_dict, lang_mono_dict, lang_sh_dict).
        Each language uses its own matched EN set so delta_XL and delta_Mono
        compare the identical questions across languages.
        """
        lang_xl:   dict[str, float] = {}
        lang_mono: dict[str, float] = {}
        lang_sh:   dict[str, float] = {}
        for lg in args.langs:
            mono_recs, xl_recs = paired_by_lang[lg]
            lang_xl[lg]   = _measure_delta_nll(model, tokenizer, xl_recs,
                                               "two_hop", "two_hop_pred", lg,
                                               layer, head, mean_val, mask_mgr_)
            lang_mono[lg] = _measure_delta_nll(model, tokenizer, mono_recs,
                                               "two_hop", "two_hop_pred", "en",
                                               layer, head, mean_val, mask_mgr_)
            lang_sh[lg]   = _measure_delta_nll(model, tokenizer, sh_samples_by_lang[lg],
                                               "second_hop", "second_hop_pred", lg,
                                               layer, head, mean_val, mask_mgr_)
        return lang_xl, lang_mono, lang_sh

    with HeadMaskManager(model, n_layers, n_heads, head_dim) as mask_mgr:
        pbar_cand = tqdm(candidates, desc="[General] ablation", unit="head")
        for layer, head in pbar_cand:
            mean_val = mean_vals.get((layer, head), 0.0)
            key = f"({layer},{head})"

            lang_xl, lang_mono, lang_sh = _deltas_per_lang(layer, head, mean_val, mask_mgr)
            xl_vals  = list(lang_xl.values())
            mean_xl  = statistics.mean(xl_vals)
            min_xl   = min(xl_vals)
            cv_xl    = (statistics.stdev(xl_vals) / (abs(mean_xl) + 1e-8)
                        if len(xl_vals) > 1 else 0.0)

            scores[key] = {
                "layer": layer, "head": head,
                "lang_delta_XL":   lang_xl,
                "lang_delta_mono": lang_mono,
                "lang_delta_sh":   lang_sh,
                "mean_delta_XL":   mean_xl,
                "min_delta_XL":    min_xl,
                "max_delta_mono":  max(lang_mono.values()),
                "max_delta_sh":    max(lang_sh.values()),
                "cv_delta_XL":     cv_xl,
            }
            pbar_cand.set_postfix(min_dXL=f"{min_xl:.4f}", cv=f"{cv_xl:.3f}")
            logger.debug("[General] (%d,%d)  min_dXL=%.4f  max_dMono=%.4f  max_dSH=%.4f",
                         layer, head, min_xl,
                         scores[key]["max_delta_mono"], scores[key]["max_delta_sh"])


    result = {
        "model": args.model_short,
        "n_candidates": len(candidates),
        "heads": scores,
    }
    save_json(result, scores_path)
    logger.info("[General] Saved general ablation scores → %s", scores_path)

    _apply_ablation_threshold_general(args, logger)


# ---------------------------------------------------------------------------
# Threshold application (re-runnable without GPU)
# ---------------------------------------------------------------------------

def _apply_ablation_threshold_specific(args: argparse.Namespace, logger) -> None:
    """Re-apply ablation threshold for all languages and save filtered_heads.json."""
    for lang in args.langs:
        scores_path   = _out_dir(args) / "stage2_ablation" / lang / "ablation_scores.json"
        filtered_path = _out_dir(args) / "stage2_ablation" / lang / "filtered_heads.json"
        if not scores_path.exists():
            continue

        data = json.loads(scores_path.read_text())
        passed = [
            v for v in data["heads"].values()
            if v["delta_XL"] > 0   # ablation must degrade XL 2-hop; negative = suppressive/irrelevant
        ]
        # Rank by delta_XL descending (importance to XL 2-hop)
        passed.sort(key=lambda x: x["delta_XL"], reverse=True)
        result = {
            "lang": lang, "model": args.model_short,
            "candidates_in":  data["n_candidates"],
            "candidates_out": len(passed),
            "heads": passed,
        }
        save_json(result, filtered_path)
        logger.info("[%s] Stage-2 (delta_XL>0): %d/%d passed → %s",
                    lang, len(passed), data["n_candidates"], filtered_path)


def _apply_ablation_threshold_general(args: argparse.Namespace, logger) -> None:
    scores_path   = _out_dir(args) / "stage2_ablation" / "general" / "ablation_scores.json"
    filtered_path = _out_dir(args) / "stage2_ablation" / "general" / "filtered_heads.json"
    if not scores_path.exists():
        return

    data = json.loads(scores_path.read_text())
    # Pass criterion: min(delta_XL across all langs) > 0
    # Every language must see degradation when the head is ablated.
    passed = [
        v for v in data["heads"].values()
        if v["min_delta_XL"] > 0
    ]
    # Rank by min_delta_XL descending (most consistent XL importance first)
    passed.sort(key=lambda x: x["min_delta_XL"], reverse=True)
    result = {
        "model": args.model_short,
        "candidates_in":  data["n_candidates"],
        "candidates_out": len(passed),
        "heads": passed,
    }
    save_json(result, filtered_path)
    logger.info("[General] Stage-2 (min_delta_XL>0): %d/%d passed → %s",
                len(passed), data["n_candidates"], filtered_path)


# ---------------------------------------------------------------------------
# Stage 3: Patchscopes
# ---------------------------------------------------------------------------

_PATCHSCOPES_SCHEMA = "r1_noun_token_v2"
_ATTENTION_SCHEMA = "question_scoped_last_token_attention_v2"
_QUESTION_MARK_CHARS = "?？¿؟"


def _load_relation_templates(args: argparse.Namespace) -> dict[str, Any]:
    return load_json(args.templates)


def _r1_noun_for_record(
    templates: dict[str, Any],
    rec: dict,
    lang: str,
) -> str:
    relation = rec.get("r1")
    if not relation:
        raise KeyError(f"Record is missing r1: {rec.get('id', rec.get('hop_id'))}")
    if relation not in templates:
        raise KeyError(f"Missing relation template for r1={relation!r}")
    if lang not in templates[relation]:
        raise KeyError(f"Missing {lang!r} template for r1={relation!r}")

    template = templates[relation][lang]
    for key in ("r_noun", "r_phrase", "embed"):
        value = template.get(key)
        if value:
            return str(value)
    raise KeyError(f"Template for r1={relation!r}, lang={lang!r} has no r1 noun field")


def _find_text_span(haystack: str, needle: str) -> tuple[int, int]:
    start = haystack.find(needle)
    if start < 0:
        start = haystack.lower().find(needle.lower())
    if start < 0:
        raise ValueError(f"Cannot find target text {needle!r} in prompt {haystack!r}")
    return start, start + len(needle)


def _question_content_span(question: str) -> tuple[int, int]:
    """Return the question body span, excluding edge question marks."""
    start = 0
    end = len(question)
    while start < end and (question[start].isspace() or question[start] in _QUESTION_MARK_CHARS):
        start += 1
    while end > start and (question[end - 1].isspace() or question[end - 1] in _QUESTION_MARK_CHARS):
        end -= 1
    if start == end:
        return 0, len(question)
    return start, end


def _target_r1_noun_token(
    tokenizer,
    prompt: str,
    question: str,
    r1_noun: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Tokenize prompt and return the last token in the r1_noun span.

    The hidden state at the final token of the r1_noun span is used as the
    Patchscopes source position.
    """
    question_start = prompt.find(question)
    if question_start < 0:
        question_start = prompt.find("Question:")
        question_start = question_start + len("Question:") if question_start >= 0 else 0

    local_start, local_end = _find_text_span(question, r1_noun)
    char_start = question_start + local_start
    char_end = question_start + local_end

    offsets = None
    try:
        encoded = tokenizer(
            prompt,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
    except (NotImplementedError, TypeError, ValueError):
        encoded = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")

    input_ids = encoded["input_ids"]
    token_ids = input_ids[0].tolist()

    token_span: list[int]
    if offsets:
        token_span = [
            idx for idx, (start, end) in enumerate(offsets)
            if end > char_start and start < char_end and end > start
        ]
    else:
        token_span = []

    if token_span:
        target_pos = token_span[-1]
    else:
        prefix_ids = tokenizer.encode(prompt[:char_end], add_special_tokens=True)
        target_pos = len(prefix_ids) - 1
        target_pos = max(0, min(target_pos, len(token_ids) - 1))
        token_span = [target_pos]

    target_token_ids = [token_ids[idx] for idx in token_span]
    target_info = {
        "target_kind": "r1_noun",
        "target_text": r1_noun,
        "target_char_span": [char_start, char_end],
        "target_token_selection": "last_token_of_r1_noun_span",
        "target_token_span": [token_span[0], token_span[-1]],
        "target_token_pos": target_pos,
        "target_token_id": token_ids[target_pos],
        "target_token": tokenizer.decode([token_ids[target_pos]]),
        "target_span_token_ids": target_token_ids,
        "target_span_tokens": [tokenizer.decode([tid]) for tid in target_token_ids],
    }
    return input_ids, target_info


def _question_token_info(
    tokenizer,
    prompt: str,
    question: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Tokenize prompt and identify tokens in the Question body, excluding marks."""
    raw_question_start = prompt.find(question)
    if raw_question_start < 0:
        raise ValueError(f"Cannot find question text in prompt: {question!r}")
    body_start, body_end = _question_content_span(question)
    question_start = raw_question_start + body_start
    question_end = raw_question_start + body_end

    offsets = None
    try:
        encoded = tokenizer(
            prompt,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
    except (NotImplementedError, TypeError, ValueError):
        encoded = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")

    input_ids = encoded["input_ids"]
    token_ids = input_ids[0].tolist()

    if offsets:
        question_positions = [
            idx for idx, (start, end) in enumerate(offsets)
            if end > question_start and start < question_end and end > start
        ]
    else:
        prefix_ids = tokenizer.encode(prompt[:question_start], add_special_tokens=True)
        question_prefix_ids = tokenizer.encode(prompt[:question_end], add_special_tokens=True)
        start_pos = max(0, min(len(prefix_ids), len(token_ids) - 1))
        end_pos = max(start_pos, min(len(question_prefix_ids) - 1, len(token_ids) - 1))
        question_positions = list(range(start_pos, end_pos + 1))

    if not question_positions:
        raise ValueError(f"No question tokens found in prompt: {prompt!r}")

    question_token_ids = [token_ids[pos] for pos in question_positions]
    info = {
        "question_text": question[body_start:body_end],
        "raw_question_text": question,
        "question_char_span": [question_start, question_end],
        "raw_question_char_span": [raw_question_start, raw_question_start + len(question)],
        "excluded_question_edge_chars": [
            ch for idx, ch in enumerate(question)
            if idx < body_start or idx >= body_end
        ],
        "question_token_span": [question_positions[0], question_positions[-1]],
        "question_token_positions": question_positions,
        "question_token_ids": question_token_ids,
        "question_tokens": [tokenizer.decode([tid]) for tid in question_token_ids],
        "last_token_pos": len(token_ids) - 1,
        "last_token_id": token_ids[-1],
        "last_token": tokenizer.decode([token_ids[-1]]),
    }
    return input_ids, info


def _get_hidden_state(
    model,
    input_ids: torch.Tensor,
    layer: int,
    position: int,
) -> torch.Tensor:
    """Extract the full residual stream at (layer, position).

    Returns a 1-D tensor of shape (hidden_dim,) on CPU.
    """
    hidden: list[torch.Tensor] = []

    def _hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            h = outputs[0]
        else:
            h = outputs
        hidden.append(h.detach().cpu())

    try:
        layer_module = model.model.layers[layer]
    except (AttributeError, IndexError):
        layer_module = model.layers[layer]

    hook_handle = layer_module.register_forward_hook(_hook)
    with torch.no_grad():
        model(input_ids=input_ids)
    hook_handle.remove()

    return hidden[0][0, position, :]  # [hidden_dim]


def _question_scoped_last_token_attention(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    layer: int,
    head: int,
    token_info: dict[str, Any],
) -> dict[str, Any]:
    """Return the final-token attention row restricted to Question-text tokens."""
    attn_weights: list[torch.Tensor] = []

    def _hook(module, inputs, outputs):
        if isinstance(outputs, tuple) and len(outputs) > 1 and outputs[1] is not None:
            attn_weights.append(outputs[1].detach().cpu())

    try:
        layer_module = model.model.layers[layer].self_attn
    except (AttributeError, IndexError):
        layer_module = model.layers[layer].self_attn

    hook_handle = layer_module.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, output_attentions=True)
    finally:
        hook_handle.remove()

    if not attn_weights:
        raise RuntimeError(f"No attention weights captured for layer={layer}, head={head}")

    row = attn_weights[0][0, head, -1, :]
    positions = [
        pos for pos in token_info["question_token_positions"]
        if pos < row.shape[0]
    ]
    if not positions:
        raise ValueError("Question token positions are outside the captured attention row")

    token_ids = input_ids[0].detach().cpu().tolist()
    scoped = []
    for pos in positions:
        token_id = token_ids[pos]
        scoped.append({
            "pos": pos,
            "token_id": token_id,
            "token": tokenizer.decode([token_id]),
            "attn_weight": float(row[pos].item()),
        })

    top_peaks = [
        {
            "rank": rank,
            "pos": item["pos"],
            "token_id": item["token_id"],
            "token": item["token"],
            "attn_weight": item["attn_weight"],
        }
        for rank, item in enumerate(
            sorted(scoped, key=lambda item: item["attn_weight"], reverse=True)[:2],
            start=1,
        )
    ]
    peak = top_peaks[0]
    second_peak = top_peaks[1] if len(top_peaks) > 1 else None
    return {
        "last_token_pos": token_info["last_token_pos"],
        "last_token_id": token_info["last_token_id"],
        "last_token": token_info["last_token"],
        "question_text": token_info["question_text"],
        "raw_question_text": token_info["raw_question_text"],
        "question_char_span": token_info["question_char_span"],
        "raw_question_char_span": token_info["raw_question_char_span"],
        "excluded_question_edge_chars": token_info["excluded_question_edge_chars"],
        "question_token_span": token_info["question_token_span"],
        "question_tokens": scoped,
        "top_question_peaks": top_peaks,
        "peak_question_pos": peak["pos"],
        "peak_question_token_id": peak["token_id"],
        "peak_question_token": peak["token"],
        "peak_question_attn_weight": peak["attn_weight"],
        "second_peak_question_pos": None if second_peak is None else second_peak["pos"],
        "second_peak_question_token_id": None if second_peak is None else second_peak["token_id"],
        "second_peak_question_token": None if second_peak is None else second_peak["token"],
        "second_peak_question_attn_weight": None if second_peak is None else second_peak["attn_weight"],
    }


def _patchscopes_probe(
    model,
    tokenizer,
    v: torch.Tensor,
    target_layer: int,
    gen_len: int,
    n_runs: int,
    e2_label_en: str,
    e2_label_lang: str,
) -> tuple[bool, float, list[dict[str, Any]]]:
    """Inject hidden state v into the decode probe and check for e2.

    Returns
    -------
    (success, encoding_rate, generations)
        success        : True if at least one run decoded e2
        encoding_rate  : fraction of runs that decoded e2
        generations    : generated text and match info for each run
    """
    probe_ids = tokenizer.encode(
        _PATCHSCOPES_PROMPT, add_special_tokens=True, return_tensors="pt"
    ).to(v.device if v.device.type != "cpu" else next(model.parameters()).device)

    # Find position of "x" (last token before ":")
    x_token_id = tokenizer.encode("x", add_special_tokens=False)[-1]
    token_list  = probe_ids[0].tolist()
    try:
        x_pos = len(token_list) - 1 - token_list[::-1].index(x_token_id)
    except ValueError:
        x_pos = probe_ids.shape[1] - 1

    # Build injection hook
    # Keep target_v on CPU and move to the target layer's device inside the hook.
    # This avoids device mismatch when layers are spread across multiple GPUs.
    target_v_cpu = v.cpu()

    def _inject_hook(module, inputs, outputs):
        # During KV-cache generation (seq_len == 1), x_pos is out of range — skip.
        if isinstance(outputs, tuple):
            h = outputs[0].clone()
            if x_pos < h.shape[1]:
                h[0, x_pos, :] = target_v_cpu.to(device=h.device, dtype=h.dtype)
            return (h,) + outputs[1:]
        else:
            out = outputs.clone()
            if x_pos < out.shape[1]:
                out[0, x_pos, :] = target_v_cpu.to(device=out.device, dtype=out.dtype)
            return out

    try:
        layer_module = model.model.layers[target_layer]
    except (AttributeError, IndexError):
        layer_module = model.layers[target_layer]

    # Filter empty strings — "" is always a substring of any generated text.
    labels: list[str] = []
    for label in (e2_label_en, e2_label_lang):
        if label and label not in labels:
            labels.append(label)
    labels_lower = [(label, label.lower()) for label in labels]
    if not labels_lower:
        return False, 0.0, []

    successes = 0
    generations: list[dict[str, Any]] = []
    for run_idx in range(n_runs):
        attention_mask = torch.ones_like(probe_ids)
        hook_handle = layer_module.register_forward_hook(_inject_hook)
        try:
            with torch.no_grad():
                out_ids = model.generate(
                    probe_ids,
                    attention_mask=attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens=gen_len,
                    do_sample=(n_runs > 1),  # greedy if single run; sampling for multiple runs
                    temperature=1.0,
                )
        finally:
            hook_handle.remove()

        generated = tokenizer.decode(
            out_ids[0, probe_ids.shape[1]:], skip_special_tokens=True
        )
        generated_lower = generated.lower()
        matched_labels = [
            label for label, label_lower in labels_lower
            if label_lower in generated_lower
        ]
        matched = bool(matched_labels)
        if matched:
            successes += 1
        generations.append(
            {
                "run_idx": run_idx,
                "generated_text": generated,
                "matched": matched,
                "matched_labels": matched_labels,
            }
        )

    encoding_rate = successes / n_runs
    return successes > 0, encoding_rate, generations


def run_patchscopes(
    args: argparse.Namespace,
    model,
    tokenizer,
    logger,
) -> None:
    n_layers = _get_num_layers(model)
    target_layer = int(n_layers * 2 / 3)
    templates = _load_relation_templates(args)
    logger.info("Patchscopes target layer: %d / %d", target_layer, n_layers)

    for lang in args.langs:
        out_lang_dir = _out_dir(args) / "stage3_patchscopes" / lang
        out_lang_dir.mkdir(parents=True, exist_ok=True)
        scores_path   = out_lang_dir / "patchscopes_scores.json"

        if scores_path.exists():
            existing = json.loads(scores_path.read_text())
            if existing.get("patchscopes_schema") == _PATCHSCOPES_SCHEMA:
                logger.info("[%s] Patchscopes scores already exist — skipping.", lang)
                continue
            logger.info("[%s] Existing Patchscopes scores use an old schema — recomputing.", lang)

        # Load stage-2 filtered heads
        filtered_path = _out_dir(args) / "stage2_ablation" / lang / "filtered_heads.json"
        if not filtered_path.exists():
            logger.warning("[%s] Stage-2 filtered_heads.json not found — skipping.", lang)
            continue

        stage2 = json.loads(filtered_path.read_text())
        candidates = [(h["layer"], h["head"]) for h in stage2["heads"]]
        if not candidates:
            logger.info("[%s] No stage-2 heads — skipping patchscopes.", lang)
            continue

        logger.info("[%s] Patchscopes: %d candidate heads", lang, len(candidates))

        lang_records = _load_filtered_by_lang(args, lang)
        xl_samples   = _sample_correct_both(lang_records, args.patchscopes_n, args.seed)
        logger.info("[%s] XL-2H samples: %d", lang, len(xl_samples))

        device   = next(model.parameters()).device

        scores: dict[str, dict] = {}
        pbar_cand = tqdm(candidates, desc=f"[{lang}] patchscopes", unit="head")
        for layer, head in pbar_cand:
            key = f"({layer},{head})"
            successes = 0
            total = 0
            record_results: list[dict] = []

            for rec_idx, rec in enumerate(tqdm(xl_samples, desc=f"  L{layer}H{head} records", unit="rec", leave=False)):
                question = rec["prompts"]["two_hop"]
                prompt = wrap_prompt(question, lang)
                r1_noun = _r1_noun_for_record(templates, rec, lang)
                input_ids, target_info = _target_r1_noun_token(
                    tokenizer, prompt, question, r1_noun
                )
                input_ids = input_ids.to(device)

                v = _get_hidden_state(
                    model, input_ids, layer, target_info["target_token_pos"]
                )

                e2_label_en   = rec.get("e2_label_en", rec.get("e2", {}).get("label_en", ""))
                e2_label_lang = rec.get("e2", {}).get("label", e2_label_en)

                success, run_encoding_rate, generations = _patchscopes_probe(
                    model, tokenizer, v,
                    target_layer=target_layer,
                    gen_len=args.patchscopes_gen_len,
                    n_runs=args.patchscopes_n_runs,
                    e2_label_en=e2_label_en,
                    e2_label_lang=e2_label_lang,
                )
                successes += int(success)
                total += 1
                record_results.append({
                    "record_idx": rec_idx,
                    "hop_id": rec.get("hop_id", rec.get("id", rec_idx)),
                    "prompt": prompt,
                    "r1": rec.get("r1"),
                    "r2": rec.get("r2"),
                    "e2_label_en": e2_label_en,
                    "e2_label_lang": e2_label_lang,
                    **target_info,
                    "success": success,
                    "run_encoding_rate": run_encoding_rate,
                    "generations": generations,
                })

            encoding_rate = successes / total if total > 0 else 0.0
            scores[key] = {
                "layer": layer, "head": head,
                "encoding_rate": encoding_rate,
                "successes": successes,
                "total": total,
                "target_token_source": "r1_noun",
                "target_token_selection": "last_token_of_r1_noun_span",
                "record_results": record_results,
            }
            pbar_cand.set_postfix(enc_rate=f"{encoding_rate:.3f}")
            logger.debug("[%s] %s  encoding_rate=%.3f", lang, key, encoding_rate)

        result = {
            "lang": lang, "model": args.model_short,
            "patchscopes_schema": _PATCHSCOPES_SCHEMA,
            "target_layer": target_layer,
            "decode_probe_prompt": _PATCHSCOPES_PROMPT,
            "target_token_source": "r1_noun",
            "target_token_selection": "last_token_of_r1_noun_span",
            "n_candidates": len(candidates),
            "heads": scores,
        }
        save_json(result, scores_path)
        logger.info("[%s] Saved patchscopes scores → %s", lang, scores_path)

    _apply_patchscopes_success_filter(args, logger)


def _apply_patchscopes_success_filter(args: argparse.Namespace, logger) -> None:
    """Keep only heads that actually generated e2 in Patchscopes."""
    for lang in args.langs:
        scores_path = _out_dir(args) / "stage3_patchscopes" / lang / "patchscopes_scores.json"
        filtered_path = _out_dir(args) / "stage3_patchscopes" / lang / "filtered_heads.json"
        if not scores_path.exists():
            continue

        data = json.loads(scores_path.read_text())
        if data.get("patchscopes_schema") != _PATCHSCOPES_SCHEMA:
            logger.warning("[%s] Patchscopes scores use an old schema — rerun --stage patchscopes.", lang)
            continue

        passed = [
            v for v in data["heads"].values()
            if int(v.get("successes", 0)) > 0
        ]
        passed.sort(
            key=lambda x: (x.get("encoding_rate", 0.0), x.get("successes", 0)),
            reverse=True,
        )
        result = {
            "lang": lang,
            "model": args.model_short,
            "patchscopes_schema": data.get("patchscopes_schema"),
            "filter_criterion": "successes > 0",
            "candidates_in": data["n_candidates"],
            "candidates_out": len(passed),
            "target_token_source": data.get("target_token_source", "r1_noun"),
            "target_token_selection": data.get(
                "target_token_selection", "last_token_of_r1_noun_span"
            ),
            "heads": passed,
        }
        save_json(result, filtered_path)
        logger.info("[%s] Patchscopes success filter: %d/%d passed → %s",
                    lang, len(passed), data["n_candidates"], filtered_path)


# ---------------------------------------------------------------------------
# Stage 2B: Question-scoped attention diagnostics
# ---------------------------------------------------------------------------

def run_question_attention(
    args: argparse.Namespace,
    model,
    tokenizer,
    logger,
) -> None:
    for lang in args.langs:
        out_lang_dir = _out_dir(args) / "stage2.5_attention" / lang
        out_lang_dir.mkdir(parents=True, exist_ok=True)
        heads_path = _out_dir(args) / "stage2_ablation" / lang / "filtered_heads.json"
        attn_path = out_lang_dir / "question_attn_weights.json"

        if not heads_path.exists():
            logger.warning("[%s] Stage-2 filtered_heads.json not found — skipping attention.", lang)
            continue

        filtered = json.loads(heads_path.read_text())
        heads = [(h["layer"], h["head"]) for h in filtered.get("heads", [])]
        if not heads:
            logger.info("[%s] No Stage-2 heads — skipping attention.", lang)
            continue

        lang_records = _load_filtered_by_lang(args, lang)
        xl_samples = _sample_correct_both(lang_records, args.patchscopes_n, args.seed)
        logger.info("[%s] Question-scoped attention: %d heads, %d samples",
                    lang, len(heads), len(xl_samples))

        device = next(model.parameters()).device
        heads_data: dict[str, Any] = {}

        for layer, head in tqdm(heads, desc=f"[{lang}] question-attn", unit="head"):
            key = f"({layer},{head})"
            records: list[dict[str, Any]] = []

            for rec_idx, rec in enumerate(tqdm(xl_samples, desc=f"  L{layer}H{head} records", unit="rec", leave=False)):
                question = rec["prompts"]["two_hop"]
                prompt = wrap_prompt(question, lang)
                input_ids, token_info = _question_token_info(tokenizer, prompt, question)
                input_ids = input_ids.to(device)

                attn_info = _question_scoped_last_token_attention(
                    model, tokenizer, input_ids, layer, head, token_info
                )
                records.append({
                    "record_idx": rec_idx,
                    "hop_id": rec.get("hop_id", rec.get("id", rec_idx)),
                    "prompt": prompt,
                    "r1": rec.get("r1"),
                    "r2": rec.get("r2"),
                    **attn_info,
                })

            heads_data[key] = {
                "layer": layer,
                "head": head,
                "total": len(records),
                "records": records,
            }

        result = {
            "lang": lang,
            "model": args.model_short,
            "attention_schema": _ATTENTION_SCHEMA,
            "head_source": "stage2_ablation/filtered_heads.json",  # heads sourced from stage2
            "attention_query": "last_input_token",
            "attention_scope": "question_text_without_edge_question_marks",
            "n_heads": len(heads),
            "n_samples": len(xl_samples),
            "heads": heads_data,
        }
        save_json(result, attn_path)
        logger.info("[%s] Saved question-scoped attention weights → %s", lang, attn_path)


# ---------------------------------------------------------------------------
# Aggregate final output
# ---------------------------------------------------------------------------

def build_final_output(args: argparse.Namespace, logger) -> None:
    out = _out_dir(args)

    final: dict[str, Any] = {
        "model": args.model_short,
        "step4_formula": args.step4_formula,
        "pipeline": {
            "stage1_bhs": "completed (see step4 output)",
            "stage2_ablation": {},
            "stage2_question_attention": {},
            "stage3_patchscopes": {},
        },
        "final_bridge_heads": {
            "general": [],
            "specific": {},
        },
    }

    # General
    gen_path = out / "stage2_ablation" / "general" / "filtered_heads.json"
    if gen_path.exists():
        g = json.loads(gen_path.read_text())
        final["pipeline"]["stage2_ablation"]["general"] = {
            "candidates_in":  g["candidates_in"],
            "candidates_out": g["candidates_out"],
        }
        final["final_bridge_heads"]["general"] = [
            (h["layer"], h["head"]) for h in g["heads"]
        ]

    # Specific per language
    for lang in args.langs:
        # Stage 2 summary
        s2_path = out / "stage2_ablation" / lang / "filtered_heads.json"
        if s2_path.exists():
            s2 = json.loads(s2_path.read_text())
            final["pipeline"]["stage2_ablation"][lang] = {
                "candidates_in":  s2["candidates_in"],
                "candidates_out": s2["candidates_out"],
                "heads": {
                    f"({h['layer']},{h['head']})": {
                        k: v for k, v in h.items() if k not in ("layer", "head")
                    }
                    for h in s2["heads"]
                },
            }

        # Stage 3 summary
        s3_path = out / "stage3_patchscopes" / lang / "filtered_heads.json"
        if s3_path.exists():
            s3 = json.loads(s3_path.read_text())
            final["pipeline"]["stage3_patchscopes"][lang] = {
                "candidates_in":  s3["candidates_in"],
                "candidates_out": s3["candidates_out"],
                "filter_criterion": s3.get("filter_criterion", "successes > 0"),
                "target_token_source": s3.get("target_token_source", "r1_noun"),
                "target_token_selection": s3.get(
                    "target_token_selection", "last_token_of_r1_noun_span"
                ),
                "heads": {
                    f"({h['layer']},{h['head']})": {"encoding_rate": h["encoding_rate"]}
                    for h in s3["heads"]
                },
            }
            final["final_bridge_heads"]["specific"][lang] = [
                (h["layer"], h["head"]) for h in s3["heads"]
            ]
        elif s2_path.exists():
            # Patchscopes not yet run: use stage-2 result
            final["final_bridge_heads"]["specific"][lang] = [
                (h["layer"], h["head"]) for h in s2["heads"]
            ]

        attn_path = out / "stage2.5_attention" / lang / "question_attn_weights.json"
        if attn_path.exists():
            attn = json.loads(attn_path.read_text())
            final["pipeline"]["stage2_question_attention"][lang] = {
                "n_heads": attn.get("n_heads"),
                "n_samples": attn.get("n_samples"),
                "attention_scope": attn.get("attention_scope"),
                "attention_query": attn.get("attention_query"),
                "head_source": attn.get("head_source"),
            }

    save_json(final, out / "final_bridge_heads.json")
    logger.info("Saved final_bridge_heads.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    logger = setup_logging(f"step5_BHS_ablation_{args.model_short}", LOG_DIR)
    logger.info("Step 5 — stage=%s  model=%s  langs=%s",
                args.stage, args.model_short, args.langs)

    # Threshold-only: no model needed
    if args.stage == "threshold":
        _apply_ablation_threshold_specific(args, logger)
        _apply_ablation_threshold_general(args, logger)
        _apply_patchscopes_success_filter(args, logger)
        build_final_output(args, logger)
        return

    # All other stages require the model
    if not args.model:
        raise ValueError("--model is required for stages: calibrate, ablation, patchscopes, attention, all.")

    logger.info("Loading model: %s", args.model)
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        device="auto",
        torch_dtype=args.torch_dtype,
        hf_token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
        attn_implementation="eager",
    )

    run_calibrate_ = args.stage in ("calibrate", "all")
    run_ablation_  = args.stage in ("ablation", "all")
    run_patches_   = args.stage in ("patchscopes", "all")
    run_attention_ = args.stage in ("attention", "all")

    if run_calibrate_:
        logger.info("=== Stage: Calibrate ===")
        run_calibrate(args, model, tokenizer, logger)

    if run_ablation_:
        logger.info("=== Stage: Ablation ===")
        mean_vals  = load_calibration_means(args)
        head_sets  = load_step4_candidates(args, logger)
        run_ablation_specific(args, model, tokenizer, head_sets, mean_vals, logger)
        run_ablation_general(args, model, tokenizer, head_sets, mean_vals, logger)

    if run_attention_:
        logger.info("=== Stage: Question-Scoped Attention ===")
        run_question_attention(args, model, tokenizer, logger)

    if run_patches_:
        logger.info("=== Stage: Patchscopes ===")
        run_patchscopes(args, model, tokenizer, logger)

    build_final_output(args, logger)
    logger.info("Step 5 complete.")


if __name__ == "__main__":
    main()
