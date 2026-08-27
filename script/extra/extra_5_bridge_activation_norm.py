"""Extra 5: Bridge-Head activation-norm comparison, success vs failure cases.

Hypothesis
----------
If the Bridge Heads (general + language-specific) causally mediate the retrieval
of the intermediate bridge entity (e2), then at the *bridge position* they should
be MORE active on cases the model gets right than on cases it gets wrong. We test
this by comparing the per-head activation norm between success and failure cases
for the exact same BRH set.

Measurement positions (both captured in ONE forward per case)
--------------------------------------------------------------
  bridge : last token of the r1_noun span -- the first-hop -> second-hop
           transition token where e2 is encoded (step5's Patchscopes source;
           we reuse ``_r1_noun_for_record`` + ``_target_r1_noun_token``
           verbatim). Tests whether the bridge signal is *generated*.
  answer : final prompt token (the 'Answer:' readout / next-token position).
           Tests whether the signal is *routed to and used at* the position
           where the answer is produced. A success>failure gap here with a
           null at bridge = "the value is generated but weakly transported"
           (effective under-weighting, not under-generation).

What "activation norm" means
----------------------------
The L2 norm of the per-head slice entering ``o_proj`` (the head's output vector
before W_O), taken at the bridge position. This is the same slice
``head_hooks.collect_mean_head_outputs`` captures, but L2 instead of abs-mean and
at one position instead of averaged.

Success / failure cases (reuses Extra 4)
----------------------------------------
Extra 5 runs AFTER Extra 4 and reuses its A100-verified subsets:
    stage3_4/data/correct100_specific_{lang}.json     -> success (specific)
    stage3_4/data/incorrect50_specific_{lang}.json    -> failure (specific)
    stage3_4/data/both_correct100_general_{lang}.json -> success (general)
    stage3_4/data/en_correct_incorrect50_general_{lang}.json -> failure (general)
If a subset is missing it is created via ``extra4.get_or_select_subset`` (same
code path). We use n_success = n_failure = 50 (balanced). Selected IDs are mapped
back to the FULL filtered records so r1 / question are available for positioning.

Groups (both shown)
-------------------
  specific : per language, heads = specific[lang], on per-lang two-hop cases.
  general  : per language AND pooled, heads = general, on cross-lingual cases.

Output layout (output/extra/extra5_{model}/)
--------------------------------------------
  general/   per_head_norms.{csv,jsonl}, group_summary.{csv,jsonl}
  specific/  per_head_norms.{csv,jsonl}, group_summary.{csv,jsonl}
  data/      per_case_norms.jsonl        (raw: one row per case, norms as a dict)
             per_case_norms_long.{csv,jsonl}  (long: one row per case x head)
  extra5_summary.json

Usage
-----
python script/extra/extra_5_bridge_activation_norm.py \
    --model-short llama31_70 --model meta-llama/Llama-3.1-70B \
    --langs ko zh ja es --stage all
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"

import sys
sys.path.insert(0, str(PROJECT_ROOT / "script"))
sys.path.insert(0, str(PROJECT_ROOT / "script" / "extra"))

from utils.common import load_json, save_json, set_seed, setup_logging
from utils.model_utils import load_model_and_tokenizer, _model_input_device
from utils.prompt_utils import wrap_prompt
from utils.head_hooks import (
    HeadMaskManager,
    _get_o_proj,
    _get_head_dim,
    _get_num_heads,
    _get_num_layers,
)

# Reuse overlapping code verbatim.
import step5_BridgeHead_Ablation_Patchscopes as s5   # bridge-position extraction
import step6_Bridge_Head_validation as s6            # bridge-head loading, saving
import extra_4_amplification_general_perf as e4      # A100 subset selection, loaders


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extra 5: BRH activation-norm comparison (success vs failure).")
    p.add_argument("--model-short", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--langs", nargs="+", default=["ko", "zh", "ja", "es"])
    p.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--cross-lingual-root", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--step5-root", type=Path,
                   default=PROJECT_ROOT / "output" / "step5_BridgeHead_Ablation_Patchscopes")
    p.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "output" / "extra",
                   help="Results go to {output-root}/extra5_{model_short}/.")
    p.add_argument("--extra4-root", type=Path, default=PROJECT_ROOT / "output" / "extra",
                   help="Where extra4_{model_short}/stage3_4/data lives (A100 subsets).")
    p.add_argument("--templates", type=Path,
                   default=PROJECT_ROOT / "script" / "config" / "relation_templates.json",
                   help="Relation templates used to locate the r1_noun (bridge) token.")
    p.add_argument("--stage", default="all", choices=["general", "specific", "all"])

    # Success / failure counts actually compared
    p.add_argument("--n-success", type=int, default=50)
    p.add_argument("--n-failure", type=int, default=50)

    # Extra-4 selection reuse (only used if the subset cache is missing)
    p.add_argument("--n-correct", type=int, default=100,
                   help="Matches extra4 subset filename (correct{N}_specific_..).")
    p.add_argument("--n-incorrect", type=int, default=50,
                   help="Matches extra4 subset filename (incorrect{N}_specific_..).")
    p.add_argument("--select-max-scan", type=int, default=600)
    p.add_argument("--n-generate-tokens", type=int, default=10)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hf-token", default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--torch-dtype", default="auto")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-head activation-norm capture at named sequence positions
# ---------------------------------------------------------------------------

POSITIONS = ("bridge", "answer")


class HeadNormCapture:
    """Capture the o_proj input at named sequence positions, per layer, so that
    each head's output-vector L2 norm can be read off after ONE forward pass.

    'bridge'  = last token of the r1_noun span (where e2 is encoded; step5's
                Patchscopes source) -- tests whether the signal is *generated*.
    'answer'  = final prompt token (the 'Answer:' readout, next-token position)
                -- tests whether the signal is *routed/used* where the answer
                is produced. Capturing both in the same forward costs nothing
                extra."""

    def __init__(self, model, layers, n_heads: int, head_dim: int) -> None:
        self.model = model
        self.layers = sorted(set(int(l) for l in layers))
        self.n_heads = n_heads
        self.head_dim = head_dim
        self._hooks: list[Any] = []
        self._positions: dict[str, int] = {}
        self._cap: dict[tuple[str, int], torch.Tensor] = {}

    def set_positions(self, positions: dict[str, int]) -> None:
        """positions: name -> sequence index, e.g. {'bridge': 12, 'answer': 41}."""
        self._positions = dict(positions)
        self._cap = {}

    def _make(self, layer_idx: int):
        def hook(module, inputs):
            x = inputs[0]
            for name, pos in self._positions.items():
                if x.dim() == 3:            # [B, S, hidden]
                    vec = x[0, pos, :]
                else:                        # [B*S, hidden] fallback (B==1)
                    vec = x[pos, :]
                self._cap[(name, layer_idx)] = vec.detach().float().cpu()
        return hook

    def __enter__(self) -> "HeadNormCapture":
        for layer_idx in self.layers:
            o_proj = _get_o_proj(self.model, layer_idx)
            if o_proj is None:
                raise RuntimeError(f"Could not locate o_proj for layer {layer_idx}.")
            self._hooks.append(o_proj.register_forward_pre_hook(self._make(layer_idx)))
        return self

    def __exit__(self, *_) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def head_norms(self, head_set: list[tuple[int, int]], pos_name: str) -> dict[tuple[int, int], float]:
        out: dict[tuple[int, int], float] = {}
        for (layer, head) in head_set:
            vec = self._cap.get((pos_name, int(layer)))
            if vec is None:
                continue
            sl = vec[head * self.head_dim:(head + 1) * self.head_dim]
            out[(int(layer), int(head))] = float(torch.linalg.vector_norm(sl).item())
        return out


# ---------------------------------------------------------------------------
# Case norm collection (one forward per case, at the r1_noun bridge position)
# ---------------------------------------------------------------------------

def _prompt_question_lang(rec: dict, cross: bool, tgt_lang: str) -> tuple[str, str, str]:
    if cross:
        lang_data = rec["langs"][tgt_lang]
        question = lang_data["prompts"]["two_hop"]
        return wrap_prompt(question, tgt_lang), question, tgt_lang
    question = rec["prompts"]["two_hop"]
    lang = rec["lang"]
    return wrap_prompt(question, lang), question, lang


def collect_case_norms(
    model, tokenizer, cap: HeadNormCapture, device,
    records: list[dict], union_heads: list[tuple[int, int]],
    templates: dict, cross: bool, tgt_lang: str, logger, tag: str,
) -> list[dict]:
    """Return [{id, lang, bridge_token, norms:{pos:{(L,H):float}}}] over records.

    One forward per case; norms captured at BOTH the bridge position (signal
    generation) and the answer readout position (signal routing/use)."""
    results: list[dict] = []
    for i, rec in enumerate(records, 1):
        prompt, question, lang = _prompt_question_lang(rec, cross, tgt_lang)
        try:
            r1_noun = s5._r1_noun_for_record(templates, rec, lang)
            input_ids, info = s5._target_r1_noun_token(tokenizer, prompt, question, r1_noun)
        except (KeyError, ValueError) as exc:
            logger.warning("  %s: skip id=%s (%s)", tag, rec.get("id"), exc)
            continue
        cap.set_positions({
            "bridge": int(info["target_token_pos"]),
            "answer": int(input_ids.shape[1] - 1),
        })
        with torch.no_grad():
            model(input_ids=input_ids.to(device))
        results.append({
            "id": rec.get("id", rec.get("hop_id")),
            "lang": lang,
            "bridge_token": info["target_token"],
            "norms": {pos: cap.head_norms(union_heads, pos) for pos in POSITIONS},
        })
        if i % 20 == 0 or i == len(records):
            logger.info("  %s: %d/%d cases", tag, i, len(records))
    return results


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _head_stats(succ_vals: list[float], fail_vals: list[float]) -> dict:
    s = np.asarray(succ_vals, dtype=float)
    f = np.asarray(fail_vals, dtype=float)
    ms = float(s.mean()) if s.size else float("nan")
    mf = float(f.mean()) if f.size else float("nan")
    ss = float(s.std(ddof=1)) if s.size > 1 else 0.0
    sf = float(f.std(ddof=1)) if f.size > 1 else 0.0
    if s.size > 1 and f.size > 1:
        pooled = math.sqrt(((s.size - 1) * ss ** 2 + (f.size - 1) * sf ** 2) / (s.size + f.size - 2))
        d = (ms - mf) / pooled if pooled > 0 else 0.0
    else:
        d = float("nan")
    try:
        p = float(scipy_stats.mannwhitneyu(s, f, alternative="two-sided").pvalue)
    except ValueError:
        p = float("nan")
    return {
        "n_success": int(s.size), "n_failure": int(f.size),
        "mean_success": ms, "mean_failure": mf,
        "std_success": ss, "std_failure": sf,
        "diff": ms - mf, "ratio": (ms / mf) if mf else float("nan"),
        "cohens_d": d, "p_value": p,
    }


def _per_head_rows(
    head_group: str, lang_label: str, head_set: list[tuple[int, int]],
    succ_results: list[dict], fail_results: list[dict], model_short: str,
) -> list[dict]:
    rows: list[dict] = []
    for position in POSITIONS:
        for (layer, head) in head_set:
            succ_vals = [r["norms"][position][(layer, head)] for r in succ_results
                         if (layer, head) in r["norms"][position]]
            fail_vals = [r["norms"][position][(layer, head)] for r in fail_results
                         if (layer, head) in r["norms"][position]]
            stats = _head_stats(succ_vals, fail_vals)
            rows.append({
                "model": model_short, "head_group": head_group, "lang": lang_label,
                "position": position, "layer": layer, "head": head, **stats,
            })
    return rows


def _group_summary_rows(head_group: str, lang_label: str, per_head_rows: list[dict],
                        model_short: str) -> list[dict]:
    out: list[dict] = []
    for position in POSITIONS:
        rows = [r for r in per_head_rows if r["position"] == position]
        if not rows:
            continue
        diffs = [r["diff"] for r in rows if not math.isnan(r["diff"])]
        ds = [r["cohens_d"] for r in rows if not math.isnan(r["cohens_d"])]
        n_heads = len(rows)
        n_up = sum(1 for r in rows if r["diff"] > 0)
        n_up_sig = sum(1 for r in rows
                       if r["diff"] > 0 and not math.isnan(r["p_value"]) and r["p_value"] < 0.05)
        out.append({
            "model": model_short, "head_group": head_group, "lang": lang_label,
            "position": position,
            "n_heads": n_heads,
            "mean_success": float(np.mean([r["mean_success"] for r in rows])),
            "mean_failure": float(np.mean([r["mean_failure"] for r in rows])),
            "mean_diff": float(np.mean(diffs)) if diffs else float("nan"),
            "mean_cohens_d": float(np.mean(ds)) if ds else float("nan"),
            "frac_heads_success_gt_failure": n_up / n_heads,
            "frac_heads_up_and_sig": n_up_sig / n_heads,
        })
    return out


# ---------------------------------------------------------------------------
# Subset resolution (reuse Extra 4's A100-verified selection)
# ---------------------------------------------------------------------------

def _resolve_ids(
    model, tokenizer, mask_mgr, subset_path: Path, source_records: list[dict],
    build_fn: Callable[[list[dict]], list[dict]], want_correct: bool, target: int,
    take: int, args: argparse.Namespace, logger, tag: str,
) -> list[str]:
    if subset_path.exists():
        items = load_json(subset_path)
        logger.info("%s: reuse extra4 subset (%d) %s", tag, len(items), subset_path.name)
    else:
        logger.info("%s: extra4 subset missing -> selecting via extra4 code.", tag)
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        source_items = build_fn(source_records)
        items = e4.get_or_select_subset(
            model, tokenizer, mask_mgr, source_items, want_correct=want_correct,
            target=target, data_path=subset_path, args=args, logger=logger, tag=tag)
    return [it["id"] for it in items][:take]


def _index_by_id(records: list[dict]) -> dict[str, dict]:
    return {r["id"]: r for r in records if "id" in r}


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_specific(
    args, model, tokenizer, cap, device, bridge_heads, templates,
    mask_mgr, extra4_data: Path, union_heads, logger,
) -> tuple[list[dict], list[dict], list[dict]]:
    per_head_all: list[dict] = []
    summary_all: list[dict] = []
    case_rows: list[dict] = []
    for lang in args.langs:
        heads = bridge_heads["specific"].get(lang, [])
        if not heads:
            logger.warning("specific: no heads for lang=%s -- skipping.", lang)
            continue
        try:
            correct_recs = e4.load_correct_records(args.input_root, args.model_short, lang)
            incorrect_recs = s6.load_incorrect_records(args.input_root, args.model_short, lang)
        except FileNotFoundError as exc:
            logger.warning("specific: data missing for lang=%s (%s) -- skipping.", lang, exc)
            continue
        id_map = _index_by_id(correct_recs) | _index_by_id(incorrect_recs)

        succ_ids = _resolve_ids(
            model, tokenizer, mask_mgr,
            extra4_data / f"correct{args.n_correct}_specific_{lang}.json",
            correct_recs, e4.build_two_hop_items, True, args.n_correct, args.n_success,
            args, logger, tag=f"select-correct-{lang}")
        fail_ids = _resolve_ids(
            model, tokenizer, mask_mgr,
            extra4_data / f"incorrect{args.n_incorrect}_specific_{lang}.json",
            incorrect_recs, e4.build_two_hop_items, False, args.n_incorrect, args.n_failure,
            args, logger, tag=f"select-incorrect-{lang}")

        succ_recs = [id_map[i] for i in succ_ids if i in id_map]
        fail_recs = [id_map[i] for i in fail_ids if i in id_map]
        logger.info("specific %s: success=%d failure=%d heads=%d",
                    lang, len(succ_recs), len(fail_recs), len(heads))

        succ_res = collect_case_norms(model, tokenizer, cap, device, succ_recs, union_heads,
                                      templates, cross=False, tgt_lang=lang, logger=logger,
                                      tag=f"specific:{lang}:success")
        fail_res = collect_case_norms(model, tokenizer, cap, device, fail_recs, union_heads,
                                      templates, cross=False, tgt_lang=lang, logger=logger,
                                      tag=f"specific:{lang}:failure")

        rows = _per_head_rows("specific", lang, heads, succ_res, fail_res, args.model_short)
        per_head_all += rows
        summary_all += _group_summary_rows("specific", lang, rows, args.model_short)
        case_rows += _case_rows("specific", lang, heads, succ_res, "success", args.model_short)
        case_rows += _case_rows("specific", lang, heads, fail_res, "failure", args.model_short)
    return per_head_all, summary_all, case_rows


def run_general(
    args, model, tokenizer, cap, device, bridge_heads, templates,
    mask_mgr, extra4_data: Path, union_heads, logger,
) -> tuple[list[dict], list[dict], list[dict]]:
    heads = bridge_heads["general"]
    per_head_all: list[dict] = []
    summary_all: list[dict] = []
    case_rows: list[dict] = []
    if not heads:
        logger.warning("general: no general heads -- skipping.")
        return per_head_all, summary_all, case_rows

    pooled_succ: list[dict] = []
    pooled_fail: list[dict] = []
    for lang in args.langs:
        try:
            both_recs = e4.load_both_correct_records(args.cross_lingual_root, args.model_short, lang)
            incorrect_recs = s6._load_cross_lingual_file(args.cross_lingual_root, args.model_short, lang)
        except FileNotFoundError as exc:
            logger.warning("general: data missing for lang=%s (%s) -- skipping.", lang, exc)
            continue
        id_map = _index_by_id(both_recs) | _index_by_id(incorrect_recs)
        build_cross = lambda recs, _l=lang: e4.build_cross_items(recs, _l)

        succ_ids = _resolve_ids(
            model, tokenizer, mask_mgr,
            extra4_data / f"both_correct{args.n_correct}_general_{lang}.json",
            both_recs, build_cross, True, args.n_correct, args.n_success,
            args, logger, tag=f"select-both-correct-{lang}")
        fail_ids = _resolve_ids(
            model, tokenizer, mask_mgr,
            extra4_data / f"en_correct_incorrect{args.n_incorrect}_general_{lang}.json",
            incorrect_recs, build_cross, False, args.n_incorrect, args.n_failure,
            args, logger, tag=f"select-incorrect-{lang}")

        succ_recs = [id_map[i] for i in succ_ids if i in id_map]
        fail_recs = [id_map[i] for i in fail_ids if i in id_map]
        logger.info("general %s: success=%d failure=%d heads=%d",
                    lang, len(succ_recs), len(fail_recs), len(heads))

        succ_res = collect_case_norms(model, tokenizer, cap, device, succ_recs, union_heads,
                                      templates, cross=True, tgt_lang=lang, logger=logger,
                                      tag=f"general:{lang}:success")
        fail_res = collect_case_norms(model, tokenizer, cap, device, fail_recs, union_heads,
                                      templates, cross=True, tgt_lang=lang, logger=logger,
                                      tag=f"general:{lang}:failure")
        pooled_succ += succ_res
        pooled_fail += fail_res

        rows = _per_head_rows("general", lang, heads, succ_res, fail_res, args.model_short)
        per_head_all += rows
        summary_all += _group_summary_rows("general", lang, rows, args.model_short)
        case_rows += _case_rows("general", lang, heads, succ_res, "success", args.model_short)
        case_rows += _case_rows("general", lang, heads, fail_res, "failure", args.model_short)

    # Pooled across languages (general heads are language-agnostic).
    if pooled_succ or pooled_fail:
        rows = _per_head_rows("general", "pooled", heads, pooled_succ, pooled_fail, args.model_short)
        per_head_all += rows
        summary_all += _group_summary_rows("general", "pooled", rows, args.model_short)
    return per_head_all, summary_all, case_rows


def _case_rows(head_group: str, lang: str, head_set, results: list[dict],
               label: str, model_short: str) -> list[dict]:
    out = []
    for r in results:
        for position in POSITIONS:
            norms = r["norms"].get(position, {})
            out.append({
                "model": model_short, "head_group": head_group, "lang": lang,
                "label": label, "position": position,
                "id": r["id"], "bridge_token": r["bridge_token"],
                "norms": {f"{L}_{H}": norms[(L, H)] for (L, H) in head_set if (L, H) in norms},
            })
    return out


def _explode_case_rows(case_rows: list[dict]) -> list[dict]:
    """Nested per-case rows -> long format: one row per (case x position x head).

    Columns: model, head_group, lang, label, position, id, bridge_token,
    layer, head, norm. Makes it trivial to later sort by norm (e.g.
    largest-norm layer/head) or contrast bridge vs answer per head."""
    long_rows: list[dict] = []
    for r in case_rows:
        for key, norm in r["norms"].items():
            layer_str, head_str = key.split("_")
            long_rows.append({
                "model": r["model"], "head_group": r["head_group"], "lang": r["lang"],
                "label": r["label"], "position": r["position"],
                "id": r["id"], "bridge_token": r["bridge_token"],
                "layer": int(layer_str), "head": int(head_str), "norm": norm,
            })
    return long_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    logger = setup_logging("extra5_bridge_activation_norm", LOG_DIR)
    logger.info("Extra 5 -- model=%s  langs=%s  stage=%s", args.model_short, args.langs, args.stage)

    model_dir = args.output_root / f"extra5_{args.model_short}"
    general_dir = model_dir / "general"
    specific_dir = model_dir / "specific"
    data_dir = model_dir / "data"
    for d in (general_dir, specific_dir, data_dir):
        d.mkdir(parents=True, exist_ok=True)
    extra4_data = args.extra4_root / f"extra4_{args.model_short}" / "stage3_4" / "data"

    bridge_heads = s6.load_final_bridge_heads(args.step5_root, args.model_short)
    templates = s5._load_relation_templates(args)
    logger.info("Bridge heads: general=%d  specific=%s",
                len(bridge_heads["general"]),
                {k: len(v) for k, v in bridge_heads["specific"].items()})

    logger.info("Loading model %s ...", args.model)
    model, tokenizer = load_model_and_tokenizer(
        args.model, device="auto", torch_dtype=args.torch_dtype,
        hf_token=args.hf_token, trust_remote_code=args.trust_remote_code)
    n_layers = _get_num_layers(model)
    n_heads = _get_num_heads(model)
    head_dim = _get_head_dim(model)
    device = _model_input_device(model)

    # Union of all BRH heads -> one capture over the layers that matter.
    union_heads = sorted(set(bridge_heads["general"])
                         | set().union(*[set(v) for v in bridge_heads["specific"].values()]))
    union_layers = {L for (L, _H) in union_heads}
    logger.info("Union BRH heads=%d across %d layers", len(union_heads), len(union_layers))

    # Mask manager only needed if extra4 subsets must be (re)created.
    mask_mgr = HeadMaskManager(model, n_layers, n_heads, head_dim)

    per_head: list[dict] = []
    summaries: list[dict] = []
    case_rows: list[dict] = []

    with HeadNormCapture(model, union_layers, n_heads, head_dim) as cap:
        if args.stage in ("specific", "all"):
            ph, sm, cr = run_specific(args, model, tokenizer, cap, device, bridge_heads,
                                      templates, mask_mgr, extra4_data, union_heads, logger)
            per_head += ph; summaries += sm; case_rows += cr
        if args.stage in ("general", "all"):
            ph, sm, cr = run_general(args, model, tokenizer, cap, device, bridge_heads,
                                     templates, mask_mgr, extra4_data, union_heads, logger)
            per_head += ph; summaries += sm; case_rows += cr

    # --- save, split by group folder ---
    spec_head = [r for r in per_head if r["head_group"] == "specific"]
    gen_head = [r for r in per_head if r["head_group"] == "general"]
    spec_sum = [r for r in summaries if r and r["head_group"] == "specific"]
    gen_sum = [r for r in summaries if r and r["head_group"] == "general"]
    if spec_head:
        s6._save_results(spec_head, specific_dir / "per_head_norms")
        s6._save_results(spec_sum, specific_dir / "group_summary")
    if gen_head:
        s6._save_results(gen_head, general_dir / "per_head_norms")
        s6._save_results(gen_sum, general_dir / "group_summary")
    if case_rows:
        # Nested (one row per case, norms as a dict).
        with (data_dir / "per_case_norms.jsonl").open("w", encoding="utf-8") as f:
            for r in case_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # Long format (one row per case x head) -> easy to sort by norm.
        long_rows = _explode_case_rows(case_rows)
        s6._save_results(long_rows, data_dir / "per_case_norms_long")
        logger.info("Per-case norms saved: %d cases (nested) + %d case x head rows (long).",
                    len(case_rows), len(long_rows))

    save_json({
        "model": args.model_short, "langs": args.langs, "stage": args.stage,
        "positions": {
            "bridge": "r1_noun last token (signal generation; step5 Patchscopes source)",
            "answer": "final prompt token, 'Answer:' readout (signal routing/use)",
        },
        "norm": "L2 of per-head o_proj input slice at bridge position",
        "n_success": args.n_success, "n_failure": args.n_failure,
        "n_general_heads": len(bridge_heads["general"]),
        "n_specific_heads": {k: len(v) for k, v in bridge_heads["specific"].items()},
        "n_per_head_rows": len(per_head), "n_case_rows": len(case_rows),
        "output_layout": {
            "general": "general/ (per_head_norms, group_summary)",
            "specific": "specific/ (per_head_norms, group_summary)",
            "data": "data/per_case_norms.jsonl",
        },
    }, model_dir / "extra5_summary.json")
    logger.info("Extra 5 complete.  per_head_rows=%d  case_rows=%d", len(per_head), len(case_rows))


if __name__ == "__main__":
    main()
