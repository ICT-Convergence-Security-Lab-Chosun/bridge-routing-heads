"""Extra 4: Does Bridge-Head amplification degrade *already-correct* / *general* performance?

Steps 6 (Stage 3/4) showed that amplifying Bridge Heads **recovers** failed cases
(base_acc = 0 by construction -> amplified_acc > 0). This experiment asks the
mirror-image question: when we apply the *same* amplification, does it also
**break** cases that were already correct, or hurt general knowledge (MMLU)?

Why this needs a fresh A100 baseline (B200 -> A100)
---------------------------------------------------
Step 6's forward direction is hardware-robust because base_acc = 0 "by
construction": 0 is 0 on any GPU. This reverse experiment is the opposite --
it is baseline-sensitive (base ~ 100%). The "correct" labels in
`correct_{model}_{lang}.json` / `en_{lang}_both_correct_{model}.json` were
produced on B200; on A100 a few of them flip at *no-intervention* baseline due
to pure numerical drift (kernel / reduction-order / cuBLAS differences). If we
inherited those labels as base_acc = 1.0 we would misattribute that drift to
amplification.

Solution: every measurement here is a **within-A100 paired delta**.
  1. We re-run each source set on A100 with NO amplification and keep only the
     cases that are *actually* correct/incorrect on this hardware -> saved to
     output/extra/extra4_{model}/data/ .
  2. Baseline and amplified conditions run in the SAME A100 session on the SAME
     items; we report acc, delta vs the in-run alpha=0 baseline, and per-item
     flips. The B200<->A100 gap loads equally on baseline and treatment and
     cancels in the difference.

Stages (4 total)
----------------
Stage 1 -- mmlu_base : MMLU accuracy with NO amplification (A100 reference).
    Samples --mmlu-max-sample questions (seeded), saves the sample and per-item
    baseline correctness to data/ for reproducible pairing.
Stage 2 -- mmlu_amp  : MMLU under amplification of General heads AND each
    language's Specific heads (one config each), across --alpha-list. Reports
    acc, delta vs the Stage-1 baseline, and per-item flips.
Stage 3 -- correct_specific (per lang) : uses correct_{model}_{lang}.json.
    Re-select 100 A100-correct + 50 A100-incorrect (from incorrect_..). Then:
      (A) correct-only  : 100 correct, base ~100%, amplify SPECIFIC heads.
      (B) mixed_50      : 50 correct + 50 incorrect, base = 50%, amplify.
Stage 4 -- correct_general (per lang) : uses en_{lang}_both_correct_{model}.json.
    Re-select 100 A100-correct + 50 A100-incorrect (from en_correct_{lang}_incorrect).
      (A) correct-only  : 100 both-correct, base ~100%, amplify GENERAL heads.
      (B) mixed_50      : 50 both-correct + 50 incorrect, base = 50%, amplify.

Three reported quantities: MMLU (delta), correct-case change (~100% ->),
mixed 50%-baseline change rate.

Output layout (output/extra/extra4_{model}/)
---------------------------------------------
  mmlu/
    data/       mmlu_sample_{model}.json, mmlu_baseline_{model}.json
    mmlu_amplification[.._random_heads].{csv,jsonl}   -- aggregate per config x alpha
    mmlu_per_question[.._random_heads].{csv,jsonl}    -- original vs amplified answer per Q
  stage3_4/
    data/       correct100_/incorrect50_/both_correct100_/en_correct_incorrect50_*.json
    correct_case_degradation[.._random_heads].{csv,jsonl}    -- aggregate per alpha
    correct_case_per_question[.._random_heads].{csv,jsonl}   -- original vs amplified answer per item
  extra4_summary[.._random_heads].json

Overlapping logic (bridge-head loading, random-head control, gold labels,
item building, alpha scaling, answer checking, result saving) is imported from
or matched to step6 verbatim, for reproducibility.

Usage
-----
# Everything, bridge heads:
python script/extra/extra_4_amplification_general_perf.py \
    --model-short llama31_70 --model meta-llama/Llama-3.1-70B \
    --langs ko zh ja es --stage all

# Random-head control (same sizes, non-bridge heads):
python script/extra/extra_4_amplification_general_perf.py \
    --model-short llama31_70 --model meta-llama/Llama-3.1-70B \
    --langs ko zh ja es --stage all --head-selection random

# MMLU only, 300 questions, 5-shot:
python script/extra/extra_4_amplification_general_perf.py \
    --model-short llama31_70 --model meta-llama/Llama-3.1-70B \
    --langs ko zh ja es --stage mmlu_base --mmlu-max-sample 300
python script/extra/extra_4_amplification_general_perf.py \
    --model-short llama31_70 --model meta-llama/Llama-3.1-70B \
    --langs ko zh ja es --stage mmlu_amp --mmlu-max-sample 300
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"

import sys
sys.path.insert(0, str(PROJECT_ROOT / "script"))

from utils.common import load_json, save_json, set_seed, setup_logging
from utils.model_utils import (
    check_answer,
    load_model_and_tokenizer,
    predict_next_tokens,
    _model_input_device,
)
from utils.prompt_utils import wrap_prompt
from utils.head_hooks import (
    HeadMaskManager,
    _get_head_dim,
    _get_num_heads,
    _get_num_layers,
)

# Import the overlapping step6 helpers verbatim (same file, same code path).
import step6_Bridge_Head_validation as s6


# ---------------------------------------------------------------------------
# CLI  (mirrors step6's relevant flags; adds MMLU + selection knobs)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extra 4: amplification effect on already-correct cases and MMLU."
    )
    p.add_argument("--model-short", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--langs", nargs="+", default=["ko", "zh", "ja", "es"],
                   help="Target languages (stage 3 per-lang; stage 4 per-target-lang).")
    p.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--cross-lingual-root", type=Path, default=PROJECT_ROOT / "data",
                   help="Root containing filtered/cross_lang/ JSON files.")
    p.add_argument("--step5-root", type=Path,
                   default=PROJECT_ROOT / "output" / "step5_BridgeHead_Ablation_Patchscopes",
                   help="Root of Step 5 output (contains {model_short}/final_bridge_heads.json).")
    p.add_argument("--output-root", type=Path,
                   default=PROJECT_ROOT / "output" / "extra",
                   help="Results go to {output-root}/extra4_{model_short}/.")

    p.add_argument("--stage", default="all",
                   choices=["mmlu_base", "mmlu_amp", "correct_specific",
                            "correct_general", "all"],
                   help="Which stage to run (default: all).")

    # Amplification (identical semantics to step6 Stage 3/4)
    p.add_argument("--alpha-list", nargs="+", type=float, default=[0.25, 0.5, 1.0],
                   help="Amplification alphas (head scaled by 1+alpha). alpha=0 baseline "
                        "is always added automatically.")
    p.add_argument("--head-selection", default="bridge", choices=["bridge", "random"],
                   help="'bridge' amplifies the real Bridge Head set; 'random' amplifies a "
                        "same-size uniformly random (layer, head) set as a control. Random "
                        "outputs are written to separate '_random_heads' files.")
    p.add_argument("--n-generate-tokens", type=int, default=10,
                   help="Max new tokens for greedy generation (two-hop stages).")

    # Case selection for stages 3 & 4
    p.add_argument("--n-correct", type=int, default=100,
                   help="A100-correct cases to select per lang (default: 100).")
    p.add_argument("--n-incorrect", type=int, default=50,
                   help="A100-incorrect cases to select per lang; also the correct-count in "
                        "the 50/50 mixed set (default: 50).")
    p.add_argument("--select-max-scan", type=int, default=600,
                   help="Max records to evaluate while searching for the target counts.")

    # MMLU
    p.add_argument("--mmlu-max-sample", type=int, default=200,
                   help="MMLU test questions to sample (0 = all 14042). Default 200.")
    p.add_argument("--mmlu-n-shot", type=int, default=5,
                   help="Few-shot examples per question (from dev split). Default 5.")
    p.add_argument("--mmlu-config", default="all",
                   help="HuggingFace cais/mmlu config name (default: all).")

    # Misc (identical to step6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hf-token", default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--torch-dtype", default="auto")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loaders (mirror step6 path logic exactly)
# ---------------------------------------------------------------------------

def load_correct_records(input_root: Path, model_short: str, lang: str) -> list[dict]:
    path = input_root / model_short / "filtered" / lang / f"correct_{model_short}_{lang}.json"
    return load_json(path)


def load_both_correct_records(root: Path, model_short: str, tgt_lang: str) -> list[dict]:
    path = root / model_short / "filtered" / "cross_lang" / f"en_{tgt_lang}_both_correct_{model_short}.json"
    if not path.exists():
        raise FileNotFoundError(f"both-correct file not found: {path}")
    return load_json(path)


# ---------------------------------------------------------------------------
# Item building (verbatim match to step6 Stage 3 / Stage 4)
# ---------------------------------------------------------------------------

def build_two_hop_items(records: list[dict]) -> list[dict]:
    """Match step6.run_scaling_amplification item construction."""
    items: list[dict] = []
    for record in records:
        labels = s6._gold_labels(record)
        if not labels:
            continue
        prompt = wrap_prompt(record["prompts"]["two_hop"], record["lang"])
        items.append({"id": record["id"], "prompt": prompt, "labels": labels})
    return items


def build_cross_items(records: list[dict], tgt_lang: str) -> list[dict]:
    """Match step6.run_transfer_accuracy item construction."""
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
    return items


# ---------------------------------------------------------------------------
# Two-hop evaluation (verbatim match to step6 Stage 3/4 inner loop)
# ---------------------------------------------------------------------------

def eval_two_hop_items(
    model, tokenizer, mask_mgr: HeadMaskManager,
    items: list[dict], amp_heads: list[tuple[int, int]], alpha: float,
    n_generate_tokens: int, logger, tag: str,
) -> tuple[int, dict[str, dict]]:
    """Run greedy generation per item with the given amplification; return
    (n_correct, {id: {"pred": text, "correct": bool}}). alpha=0 with amp_heads
    scales by (1+0)=1.0 => identity."""
    correct = 0
    per_item: dict[str, dict] = {}
    for i, item in enumerate(items, 1):
        mask_mgr.reset_masks()
        if amp_heads:
            mask_mgr.apply_head_set_scaling(amp_heads, alpha)
        pred = predict_next_tokens(model, tokenizer, item["prompt"], n_tokens=n_generate_tokens)
        mask_mgr.reset_masks()
        pred_text = pred.split("\n")[0].strip()
        ok = check_answer(pred_text, item["labels"])
        per_item[item["id"]] = {"pred": pred_text, "correct": ok}
        correct += int(ok)
        if i % 20 == 0 or i == len(items):
            logger.info("    %s alpha=%.2f  item %d/%d  acc_so_far=%.3f",
                        tag, alpha, i, len(items), correct / i)
    return correct, per_item


# ---------------------------------------------------------------------------
# A100 subset selection (the within-A100 re-filtering)
# ---------------------------------------------------------------------------

def _subset_to_records(items: list[dict]) -> list[dict]:
    return [{"id": it["id"], "prompt": it["prompt"], "labels": it["labels"]} for it in items]


def get_or_select_subset(
    model, tokenizer, mask_mgr: HeadMaskManager,
    source_items: list[dict], want_correct: bool, target: int,
    data_path: Path, args: argparse.Namespace, logger, tag: str,
) -> list[dict]:
    """Return `target` items that are actually correct/incorrect at A100 baseline.

    Cached to `data_path` so re-runs reuse the exact same subset. Selection scans
    a seeded shuffle of the source and keeps items matching `want_correct` until
    `target` is reached (or --select-max-scan is hit)."""
    if data_path.exists():
        cached = load_json(data_path)
        logger.info("%s: loaded cached subset (%d items) from %s", tag, len(cached), data_path.name)
        return cached

    items = list(source_items)
    set_seed(args.seed)
    random.shuffle(items)

    selected: list[dict] = []
    scanned = 0
    base_correct = 0
    for item in items:
        if len(selected) >= target or scanned >= args.select_max_scan:
            break
        scanned += 1
        mask_mgr.reset_masks()
        pred = predict_next_tokens(model, tokenizer, item["prompt"], n_tokens=args.n_generate_tokens)
        ok = check_answer(pred.split("\n")[0].strip(), item["labels"])
        base_correct += int(ok)
        if ok == want_correct:
            selected.append(item)
        if scanned % 20 == 0:
            logger.info("  %s: scanned=%d  base_acc=%.3f  selected=%d/%d",
                        tag, scanned, base_correct / scanned, len(selected), target)

    logger.info("%s: DONE scanned=%d  A100_base_acc=%.3f  selected=%d/%d",
                tag, scanned, base_correct / max(scanned, 1), len(selected), target)
    if len(selected) < target:
        logger.warning("%s: only found %d/%d matching cases within --select-max-scan=%d.",
                       tag, len(selected), target, args.select_max_scan)

    records = _subset_to_records(selected)
    save_json(records, data_path, metadata={
        "tag": tag, "want_correct": want_correct, "target": target,
        "scanned": scanned, "a100_base_acc": base_correct / max(scanned, 1),
        "n_selected": len(records), "seed": args.seed, "model": args.model_short,
    })
    return records


# ---------------------------------------------------------------------------
# Amplification analysis (correct-only and mixed-50) with paired flips
# ---------------------------------------------------------------------------

def run_amplification_analysis(
    model, tokenizer, args: argparse.Namespace,
    items: list[dict], amp_heads: list[tuple[int, int]],
    analysis: str, lang_key: str, lang_value: str, head_set_type: str,
    n_layers: int, n_heads: int, head_dim: int, logger,
) -> tuple[list[dict], list[dict]]:
    """Sweep [0.0] + alpha_list on one item set; return (aggregate rows,
    per-question rows). Per-question rows carry the original (alpha=0) answer
    and the amplified answer for each item, so flips are inspectable."""
    if not items:
        return [], []
    alphas = [0.0] + [a for a in args.alpha_list if a != 0.0]
    head_selection = args.head_selection
    rows: list[dict] = []
    per_q_rows: list[dict] = []
    base_map: dict[str, dict] | None = None
    base_acc = float("nan")

    with HeadMaskManager(model, n_layers, n_heads, head_dim) as mm:
        for alpha in alphas:
            correct, per_map = eval_two_hop_items(
                model, tokenizer, mm, items, amp_heads, alpha,
                args.n_generate_tokens, logger,
                tag=f"{head_set_type}:{lang_value}:{analysis}",
            )
            acc = correct / len(items)
            if alpha == 0.0:
                base_map = per_map
                base_acc = acc
            c2i = sum(1 for k, v in base_map.items()
                      if v["correct"] and not per_map.get(k, {}).get("correct", False)) if base_map else 0
            i2c = sum(1 for k, v in base_map.items()
                      if not v["correct"] and per_map.get(k, {}).get("correct", False)) if base_map else 0
            rows.append({
                "model": args.model_short,
                "stage": f"correct_{head_set_type}",
                "analysis": analysis,
                lang_key: lang_value,
                "head_set_type": head_set_type,
                "head_selection": head_selection,
                "intervention": "scaling",
                "alpha": alpha,
                "k": len(amp_heads),
                "n_items": len(items),
                "n_correct": correct,
                "accuracy": acc,
                "baseline_accuracy": base_acc,
                "delta": acc - base_acc,
                "n_correct_to_incorrect": c2i,
                "n_incorrect_to_correct": i2c,
            })
            # Per-question original vs amplified answer (baseline embedded as reference).
            if base_map is not None and alpha != 0.0:
                for k, cur in per_map.items():
                    b = base_map.get(k, {})
                    per_q_rows.append({
                        "model": args.model_short,
                        "stage": f"correct_{head_set_type}",
                        "analysis": analysis,
                        lang_key: lang_value,
                        "head_set_type": head_set_type,
                        "head_selection": head_selection,
                        "alpha": alpha,
                        "id": k,
                        "baseline_pred": b.get("pred"),
                        "baseline_correct": bool(b.get("correct")),
                        "amp_pred": cur["pred"],
                        "amp_correct": cur["correct"],
                        "answer_changed": b.get("pred") != cur["pred"],
                        "correctness_change": int(cur["correct"]) - int(bool(b.get("correct"))),
                    })
            logger.info(
                "%s %-12s %s=%-4s alpha=%.2f  acc=%.4f  base=%.4f  d=%+.4f  c->i=%d i->c=%d",
                head_set_type, analysis, lang_key, lang_value, alpha, acc, base_acc,
                acc - base_acc, c2i, i2c,
            )
    return rows, per_q_rows


def _resolve_amp_heads(
    base_heads: list[tuple[int, int]], args: argparse.Namespace,
    n_layers: int, n_heads: int, rng_key: str, logger,
) -> list[tuple[int, int]]:
    """Bridge heads, or a same-size random control (mirrors step6)."""
    if args.head_selection == "random":
        rng = random.Random(f"{args.seed}-extra4-{rng_key}")
        amp = s6.sample_random_head_set(base_heads, n_layers, n_heads, rng)
        logger.info("  %s: RANDOM head set (k=%d) instead of bridge heads.", rng_key, len(amp))
        return amp
    return base_heads


# ---------------------------------------------------------------------------
# Stage 3: correct-case degradation, SPECIFIC heads, per language
# ---------------------------------------------------------------------------

def run_correct_specific(
    lang: str, args: argparse.Namespace, model, tokenizer,
    bridge_heads: dict[str, Any], data_dir: Path,
    n_layers: int, n_heads: int, head_dim: int, logger,
) -> tuple[list[dict], list[dict]]:
    specific = bridge_heads["specific"].get(lang, [])
    if not specific:
        logger.warning("correct_specific: no specific heads for lang=%s -- skipping.", lang)
        return [], []
    amp_heads = _resolve_amp_heads(specific, args, n_layers, n_heads, f"specific-{lang}", logger)

    # --- A100 subset selection ---
    try:
        correct_src = build_two_hop_items(load_correct_records(args.input_root, args.model_short, lang))
        incorrect_src = build_two_hop_items(s6.load_incorrect_records(args.input_root, args.model_short, lang))
    except FileNotFoundError as exc:
        logger.warning("correct_specific: data missing for lang=%s (%s) -- skipping.", lang, exc)
        return [], []

    with HeadMaskManager(model, n_layers, n_heads, head_dim) as mm:
        correct100 = get_or_select_subset(
            model, tokenizer, mm, correct_src, want_correct=True, target=args.n_correct,
            data_path=data_dir / f"correct{args.n_correct}_specific_{lang}.json",
            args=args, logger=logger, tag=f"select-correct-{lang}")
        incorrect50 = get_or_select_subset(
            model, tokenizer, mm, incorrect_src, want_correct=False, target=args.n_incorrect,
            data_path=data_dir / f"incorrect{args.n_incorrect}_specific_{lang}.json",
            args=args, logger=logger, tag=f"select-incorrect-{lang}")

    rows: list[dict] = []
    per_q: list[dict] = []
    # (A) correct-only: base ~100%
    r, pq = run_amplification_analysis(
        model, tokenizer, args, correct100, amp_heads, "correct_only",
        "lang", lang, "specific", n_layers, n_heads, head_dim, logger)
    rows += r; per_q += pq
    # (B) mixed 50/50: base = 50%
    mixed = correct100[:args.n_incorrect] + incorrect50
    r, pq = run_amplification_analysis(
        model, tokenizer, args, mixed, amp_heads, "mixed_50",
        "lang", lang, "specific", n_layers, n_heads, head_dim, logger)
    rows += r; per_q += pq
    return rows, per_q


# ---------------------------------------------------------------------------
# Stage 4: cross-lingual both-correct degradation, GENERAL heads, per lang
# ---------------------------------------------------------------------------

def run_correct_general(
    lang: str, args: argparse.Namespace, model, tokenizer,
    bridge_heads: dict[str, Any], data_dir: Path,
    n_layers: int, n_heads: int, head_dim: int, logger,
) -> tuple[list[dict], list[dict]]:
    general = bridge_heads["general"]
    if not general:
        logger.warning("correct_general: no general heads -- skipping.")
        return [], []
    amp_heads = _resolve_amp_heads(general, args, n_layers, n_heads, f"general-{lang}", logger)

    # --- A100 subset selection (target-language inference) ---
    try:
        both_src = build_cross_items(
            load_both_correct_records(args.cross_lingual_root, args.model_short, lang), lang)
        incorrect_src = build_cross_items(
            s6._load_cross_lingual_file(args.cross_lingual_root, args.model_short, lang), lang)
    except FileNotFoundError as exc:
        logger.warning("correct_general: data missing for lang=%s (%s) -- skipping.", lang, exc)
        return [], []

    with HeadMaskManager(model, n_layers, n_heads, head_dim) as mm:
        correct100 = get_or_select_subset(
            model, tokenizer, mm, both_src, want_correct=True, target=args.n_correct,
            data_path=data_dir / f"both_correct{args.n_correct}_general_{lang}.json",
            args=args, logger=logger, tag=f"select-both-correct-{lang}")
        incorrect50 = get_or_select_subset(
            model, tokenizer, mm, incorrect_src, want_correct=False, target=args.n_incorrect,
            data_path=data_dir / f"en_correct_incorrect{args.n_incorrect}_general_{lang}.json",
            args=args, logger=logger, tag=f"select-incorrect-{lang}")

    rows: list[dict] = []
    per_q: list[dict] = []
    r, pq = run_amplification_analysis(
        model, tokenizer, args, correct100, amp_heads, "correct_only",
        "tgt_lang", lang, "general", n_layers, n_heads, head_dim, logger)
    rows += r; per_q += pq
    mixed = correct100[:args.n_incorrect] + incorrect50
    r, pq = run_amplification_analysis(
        model, tokenizer, args, mixed, amp_heads, "mixed_50",
        "tgt_lang", lang, "general", n_layers, n_heads, head_dim, logger)
    rows += r; per_q += pq
    return rows, per_q


# ---------------------------------------------------------------------------
# MMLU
# ---------------------------------------------------------------------------

_LETTERS = ["A", "B", "C", "D"]


def _mmlu_option_token_ids(tokenizer) -> list[int]:
    """Token id for the answer letter as it appears after 'Answer:' (leading space)."""
    ids: list[int] = []
    for letter in _LETTERS:
        enc = tokenizer.encode(" " + letter, add_special_tokens=False)
        ids.append(enc[-1])
    return ids


def _format_mmlu_example(question: str, choices: list[str], answer_idx: int | None) -> str:
    """Verbatim match to Hendrycks et al. (2021) ``format_example``.

    Non-answer form:  ``{q}\\nA. {c0}\\nB. {c1}\\nC. {c2}\\nD. {c3}\\nAnswer:``
    Answer form appends ``  {letter}\\n\\n`` so exemplars self-separate by one
    blank line when concatenated (exactly as the original repo).
    """
    s = question
    for i, choice in enumerate(choices):
        s += f"\n{_LETTERS[i]}. {choice}"
    s += "\nAnswer:"
    if answer_idx is not None:
        s += f" {_LETTERS[answer_idx]}\n\n"
    return s


def build_mmlu_prompt(subject: str, dev_examples: list[dict], test_ex: dict, n_shot: int) -> str:
    """Verbatim match to Hendrycks et al. ``gen_prompt`` + appended test example."""
    prompt = ("The following are multiple choice questions (with answers) about "
              f"{subject.replace('_', ' ')}.\n\n")
    for ex in dev_examples[:n_shot]:
        prompt += _format_mmlu_example(ex["question"], ex["choices"], ex["answer"])
    prompt += _format_mmlu_example(test_ex["question"], test_ex["choices"], None)
    return prompt


def load_mmlu(config: str, logger):
    from datasets import load_dataset
    logger.info("Loading MMLU (cais/mmlu, config=%s) ...", config)
    test = load_dataset("cais/mmlu", config, split="test")
    dev = load_dataset("cais/mmlu", config, split="dev")
    dev_by_subject: dict[str, list[dict]] = {}
    for ex in dev:
        dev_by_subject.setdefault(ex["subject"], []).append(
            {"question": ex["question"], "choices": ex["choices"], "answer": ex["answer"]})
    return test, dev_by_subject


def get_or_build_mmlu_sample(test, args: argparse.Namespace, data_path: Path, logger) -> list[dict]:
    if data_path.exists():
        cached = load_json(data_path)
        logger.info("MMLU sample: loaded %d cached questions from %s", len(cached), data_path.name)
        return cached
    n = len(test)
    if args.mmlu_max_sample <= 0 or args.mmlu_max_sample >= n:
        idxs = list(range(n))  # full canonical test set (all 14042)
    else:
        # Stratified by subject so the subsample still covers all 57 subjects
        # with counts proportional to the full test set (>=1 each).
        by_subject: dict[str, list[int]] = {}
        for i, subj in enumerate(test["subject"]):
            by_subject.setdefault(subj, []).append(i)
        set_seed(args.seed)
        idxs = []
        for subj in sorted(by_subject):
            pool = by_subject[subj]
            k = max(1, round(args.mmlu_max_sample * len(pool) / n))
            k = min(k, len(pool))
            idxs.extend(random.sample(pool, k))
        idxs.sort()
    sample = [{
        "subject": test[i]["subject"], "question": test[i]["question"],
        "choices": list(test[i]["choices"]), "answer": int(test[i]["answer"]),
    } for i in idxs]
    save_json(sample, data_path, metadata={
        "n_total_test": n, "n_sampled": len(sample), "seed": args.seed,
        "sampling": "full" if len(sample) == n else "stratified_by_subject",
        "mmlu_config": args.mmlu_config, "n_shot": args.mmlu_n_shot,
    })
    logger.info("MMLU sample: built and saved %d questions to %s", len(sample), data_path.name)
    return sample


@torch.no_grad()
def _mmlu_predict_idx(model, tokenizer, prompt: str, option_ids: list[int], device) -> int:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits[0, -1]
    opt_logits = torch.stack([logits[i] for i in option_ids])
    return int(opt_logits.argmax().item())


def score_mmlu(
    model, tokenizer, mask_mgr: HeadMaskManager, sample: list[dict],
    dev_by_subject: dict[str, list[dict]], option_ids: list[int], device,
    amp_heads: list[tuple[int, int]], alpha: float, n_shot: int, logger, tag: str,
) -> tuple[int, list[dict]]:
    """Return (n_correct, per-question records). Each record keeps the predicted
    letter so the original / amplified answer can be saved per question."""
    correct = 0
    records: list[dict] = []
    for i, ex in enumerate(sample, 1):
        mask_mgr.reset_masks()
        if amp_heads:
            mask_mgr.apply_head_set_scaling(amp_heads, alpha)
        prompt = build_mmlu_prompt(ex["subject"], dev_by_subject.get(ex["subject"], []), ex, n_shot)
        pred_idx = _mmlu_predict_idx(model, tokenizer, prompt, option_ids, device)
        mask_mgr.reset_masks()
        ok = (pred_idx == ex["answer"])
        records.append({
            "idx": i - 1, "subject": ex["subject"],
            "gold": _LETTERS[ex["answer"]], "pred": _LETTERS[pred_idx], "correct": ok,
        })
        correct += int(ok)
        if i % 50 == 0 or i == len(sample):
            logger.info("    %s alpha=%.2f  %d/%d  acc_so_far=%.3f",
                        tag, alpha, i, len(sample), correct / i)
    return correct, records


def _load_baseline_records(base_path: Path, n_expected: int, logger) -> list[dict] | None:
    """Load per-question baseline records if present, valid, and length-matching."""
    if not base_path.exists():
        return None
    loaded = load_json(base_path)
    if not loaded or not isinstance(loaded[0], dict) or "pred" not in loaded[0]:
        logger.warning("mmlu_amp: cached baseline has unexpected format -- recomputing.")
        return None
    if len(loaded) != n_expected:
        logger.warning("mmlu_amp: cached baseline length mismatch -- recomputing baseline.")
        return None
    return loaded


def run_mmlu_base(
    args: argparse.Namespace, model, tokenizer, data_dir: Path,
    n_layers: int, n_heads: int, head_dim: int, logger,
) -> list[dict]:
    test, dev_by_subject = load_mmlu(args.mmlu_config, logger)
    sample = get_or_build_mmlu_sample(test, args, data_dir / f"mmlu_sample_{args.model_short}.json", logger)
    option_ids = _mmlu_option_token_ids(tokenizer)
    device = _model_input_device(model)

    with HeadMaskManager(model, n_layers, n_heads, head_dim) as mm:
        correct, records = score_mmlu(
            model, tokenizer, mm, sample, dev_by_subject, option_ids, device,
            amp_heads=[], alpha=0.0, n_shot=args.mmlu_n_shot, logger=logger, tag="mmlu:baseline")
    acc = correct / len(sample)

    # Per-question baseline predictions (used by mmlu_amp for paired flips).
    save_json(records, data_dir / f"mmlu_baseline_{args.model_short}.json", metadata={
        "n_items": len(sample), "n_correct": correct, "accuracy": acc,
        "n_shot": args.mmlu_n_shot, "seed": args.seed,
    })
    logger.info("MMLU baseline: acc=%.4f  (%d/%d)", acc, correct, len(sample))
    return [{
        "model": args.model_short, "stage": "mmlu", "config": "baseline",
        "head_selection": args.head_selection, "alpha": 0.0, "k": 0,
        "n_items": len(sample), "n_correct": correct, "accuracy": acc,
        "baseline_accuracy": acc, "delta": 0.0,
        "n_correct_to_incorrect": 0, "n_incorrect_to_correct": 0,
    }]


def run_mmlu_amp(
    args: argparse.Namespace, model, tokenizer, bridge_heads: dict[str, Any],
    data_dir: Path, n_layers: int, n_heads: int, head_dim: int, logger,
) -> list[dict]:
    model_dir = data_dir.parent
    suffix = "_random_heads" if args.head_selection == "random" else ""

    sample_path = data_dir / f"mmlu_sample_{args.model_short}.json"
    test, dev_by_subject = load_mmlu(args.mmlu_config, logger)
    if sample_path.exists():
        sample = load_json(sample_path)
        logger.info("mmlu_amp: loaded %d cached questions from %s", len(sample), sample_path.name)
    else:
        logger.info("mmlu_amp: sample not found -- building it now (standalone run).")
        sample = get_or_build_mmlu_sample(test, args, sample_path, logger)
    option_ids = _mmlu_option_token_ids(tokenizer)
    device = _model_input_device(model)

    # In-run A100 baseline for paired flips (prefer the saved Stage-1 baseline).
    base_path = data_dir / f"mmlu_baseline_{args.model_short}.json"
    base_records = _load_baseline_records(base_path, len(sample), logger)
    with HeadMaskManager(model, n_layers, n_heads, head_dim) as mm:
        if base_records is None:
            logger.info("mmlu_amp: computing baseline in-run (no usable cached baseline).")
            base_correct, base_records = score_mmlu(
                model, tokenizer, mm, sample, dev_by_subject, option_ids, device,
                amp_heads=[], alpha=0.0, n_shot=args.mmlu_n_shot, logger=logger, tag="mmlu:baseline")
            save_json(base_records, base_path, metadata={
                "n_items": len(sample), "n_correct": base_correct,
                "accuracy": base_correct / len(sample), "n_shot": args.mmlu_n_shot,
                "seed": args.seed, "bootstrapped_by": "mmlu_amp",
            })
    base_acc = sum(int(r["correct"]) for r in base_records) / len(base_records)

    # Configs: general + each language's specific set.
    configs: list[tuple[str, list[tuple[int, int]]]] = [("general", bridge_heads["general"])]
    for lang in args.langs:
        sp = bridge_heads["specific"].get(lang, [])
        if sp:
            configs.append((f"specific_{lang}", sp))

    rows: list[dict] = []          # aggregate (per config x alpha)
    per_q_rows: list[dict] = []    # per-question: original answer + amplified answer
    for name, base_heads in configs:
        if not base_heads:
            logger.warning("mmlu_amp: no heads for config=%s -- skipping.", name)
            continue
        amp_heads = _resolve_amp_heads(base_heads, args, n_layers, n_heads, f"mmlu-{name}", logger)
        with HeadMaskManager(model, n_layers, n_heads, head_dim) as mm:
            for alpha in args.alpha_list:
                correct, amp_records = score_mmlu(
                    model, tokenizer, mm, sample, dev_by_subject, option_ids, device,
                    amp_heads, alpha, args.mmlu_n_shot, logger, tag=f"mmlu:{name}")
                acc = correct / len(sample)
                c2i = sum(1 for b, a in zip(base_records, amp_records) if b["correct"] and not a["correct"])
                i2c = sum(1 for b, a in zip(base_records, amp_records) if not b["correct"] and a["correct"])
                rows.append({
                    "model": args.model_short, "stage": "mmlu", "config": name,
                    "head_selection": args.head_selection, "alpha": alpha, "k": len(amp_heads),
                    "n_items": len(sample), "n_correct": correct, "accuracy": acc,
                    "baseline_accuracy": base_acc, "delta": acc - base_acc,
                    "n_correct_to_incorrect": c2i, "n_incorrect_to_correct": i2c,
                })
                for b, a in zip(base_records, amp_records):
                    per_q_rows.append({
                        "model": args.model_short, "config": name,
                        "head_selection": args.head_selection, "alpha": alpha,
                        "idx": a["idx"], "subject": a["subject"], "gold": a["gold"],
                        "baseline_pred": b["pred"], "baseline_correct": b["correct"],
                        "amp_pred": a["pred"], "amp_correct": a["correct"],
                        "answer_changed": b["pred"] != a["pred"],
                        "correctness_change": int(a["correct"]) - int(b["correct"]),
                    })
                logger.info("MMLU %-14s alpha=%.2f  acc=%.4f  base=%.4f  d=%+.4f  c->i=%d i->c=%d",
                            name, alpha, acc, base_acc, acc - base_acc, c2i, i2c)

    if per_q_rows:
        s6._save_results(per_q_rows, model_dir / f"mmlu_per_question{suffix}")
        logger.info("MMLU per-question predictions saved: %d rows -> mmlu_per_question%s.{csv,jsonl}",
                    len(per_q_rows), suffix)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    logger = setup_logging("extra4_amplification_general_perf", LOG_DIR)
    logger.info(
        "Extra 4 -- model=%s  langs=%s  stage=%s  head_selection=%s",
        args.model_short, args.langs, args.stage, args.head_selection,
    )

    model_dir = args.output_root / f"extra4_{args.model_short}"
    # Separate output folders for clarity: MMLU vs Stage 3/4 (correct-case).
    mmlu_dir = model_dir / "mmlu"
    mmlu_data = mmlu_dir / "data"
    stage34_dir = model_dir / "stage3_4"
    stage34_data = stage34_dir / "data"
    for d in (mmlu_data, stage34_data):
        d.mkdir(parents=True, exist_ok=True)
    suffix = "_random_heads" if args.head_selection == "random" else ""

    bridge_heads = s6.load_final_bridge_heads(args.step5_root, args.model_short)
    logger.info("Bridge heads: general=%d  specific=%s",
                len(bridge_heads["general"]),
                {k: len(v) for k, v in bridge_heads["specific"].items()})

    logger.info("Loading model %s ...", args.model)
    model, tokenizer = load_model_and_tokenizer(
        args.model, device="auto", torch_dtype=args.torch_dtype,
        hf_token=args.hf_token, trust_remote_code=args.trust_remote_code,
    )
    n_layers = _get_num_layers(model)
    n_heads = _get_num_heads(model)
    head_dim = _get_head_dim(model)

    all_mmlu: list[dict] = []
    all_correct: list[dict] = []
    all_correct_pq: list[dict] = []

    if args.stage in ("mmlu_base", "all"):
        all_mmlu += run_mmlu_base(args, model, tokenizer, mmlu_data, n_layers, n_heads, head_dim, logger)

    if args.stage in ("mmlu_amp", "all"):
        all_mmlu += run_mmlu_amp(args, model, tokenizer, bridge_heads, mmlu_data,
                                 n_layers, n_heads, head_dim, logger)

    if args.stage in ("correct_specific", "all"):
        for lang in args.langs:
            r, pq = run_correct_specific(
                lang, args, model, tokenizer, bridge_heads, stage34_data,
                n_layers, n_heads, head_dim, logger)
            all_correct += r
            all_correct_pq += pq

    if args.stage in ("correct_general", "all"):
        for lang in args.langs:
            r, pq = run_correct_general(
                lang, args, model, tokenizer, bridge_heads, stage34_data,
                n_layers, n_heads, head_dim, logger)
            all_correct += r
            all_correct_pq += pq

    # --- save results (step6-style: csv + jsonl per group), split by folder ---
    if all_mmlu:
        s6._save_results(all_mmlu, mmlu_dir / f"mmlu_amplification{suffix}")
    if all_correct:
        s6._save_results(all_correct, stage34_dir / f"correct_case_degradation{suffix}")
    if all_correct_pq:
        s6._save_results(all_correct_pq, stage34_dir / f"correct_case_per_question{suffix}")
        logger.info("Stage 3/4 per-question predictions saved: %d rows -> "
                    "stage3_4/correct_case_per_question%s.{csv,jsonl}", len(all_correct_pq), suffix)

    summary_dir = model_dir
    summary = {
        "model": args.model_short,
        "langs": args.langs,
        "stage": args.stage,
        "head_selection": args.head_selection,
        "alpha_list": args.alpha_list,
        "mmlu_max_sample": args.mmlu_max_sample,
        "mmlu_n_shot": args.mmlu_n_shot,
        "n_correct": args.n_correct,
        "n_incorrect": args.n_incorrect,
        "n_general_heads": len(bridge_heads["general"]),
        "n_specific_heads": {k: len(v) for k, v in bridge_heads["specific"].items()},
        "n_mmlu_rows": len(all_mmlu),
        "n_correct_rows": len(all_correct),
        "n_correct_per_question_rows": len(all_correct_pq),
        "output_layout": {
            "mmlu": "mmlu/ (data/, mmlu_amplification, mmlu_per_question)",
            "stage3_4": "stage3_4/ (data/, correct_case_degradation, correct_case_per_question)",
        },
    }
    save_json(summary, summary_dir / f"extra4_summary{suffix}.json")
    logger.info("Extra 4 complete.  mmlu_rows=%d  correct_rows=%d", len(all_mmlu), len(all_correct))


if __name__ == "__main__":
    main()
