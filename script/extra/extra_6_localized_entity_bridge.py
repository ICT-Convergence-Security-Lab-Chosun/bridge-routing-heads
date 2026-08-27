"""Extra 6: Does the general Bridge-Head pool survive entity localization?

Reviewer concern
----------------
Keeping source entities in English gives every non-English prompt an English
anchor. Since the paper claims *language-general* routing heads, we must check
whether the general BRH pool still emerges when entity names are localized into
the target language/script (rather than being an artifact of the shared English
anchor).

Experiment
----------
1. Rebuild data with target-language entity labels in the prompt
   (step1 --entity-prompt-label localized -> data/processed_localized).
2. Re-run the pipeline up to Stage 2 (mean-ablation filter) ONLY -- no
   Patchscopes -- under a separate namespace ``{model}_loc``:
       step2 (eval) -> step3 (filter) -> step4 (BRS) -> step5 --stage ablation
3. This script compares the Stage-2 head sets of the original vs. localized runs
   (general + per-language specific), which are already saved at
       output/step5.../{model}/stage2_ablation/{general,lang}/filtered_heads.json
       output/step5.../{model}_loc/stage2_ablation/{general,lang}/filtered_heads.json

Metrics
-------
  - Jaccard overlap of the Stage-2 head sets (general, per-lang specific).
  - Overlap significance vs. random (hypergeometric over all model heads).
  - Spearman correlation of per-head delta_XL over the shared candidate pool
    (rank stability of head importance).
  - Confounds reported alongside: 2-hop correct-sample counts (orig vs loc) and
    localization coverage per language (fraction of e1/e2 actually localized).

Interpretation
--------------
If, despite localization, the general pool re-emerges with overlap well above
chance (and stable delta_XL ranking), the general BRH are genuinely language-
general -- not an English-anchor artifact. Low overlap AT ADEQUATE coverage AND
sample counts would support the reviewer's concern; low overlap with collapsed
coverage/accuracy is a power artifact, not evidence.

This is a CPU-only analysis over existing JSON outputs (no model load).

Usage
-----
python script/extra/extra_6_localized_entity_bridge.py --models llama31_70 --langs ko zh ja es
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import stats as scipy_stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys
sys.path.insert(0, str(PROJECT_ROOT / "script"))
from utils.common import setup_logging  # noqa: E402

LOG_DIR = PROJECT_ROOT / "logs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extra 6: localized-entity robustness of the general Bridge-Head pool.")
    p.add_argument("--models", nargs="+", default=["llama31_70", "qwen25_72"],
                   help="Original model_short names (localized run = {model}{loc-suffix}).")
    p.add_argument("--loc-suffix", default="_loc")
    p.add_argument("--langs", nargs="+", default=["ko", "zh", "ja", "es"])
    p.add_argument("--step5-root", type=Path,
                   default=PROJECT_ROOT / "output" / "step5_BridgeHead_Ablation_Patchscopes")
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--localized-processed-dir", type=Path,
                   default=PROJECT_ROOT / "data" / "processed_localized")
    p.add_argument("--output-root", type=Path,
                   default=PROJECT_ROOT / "output" / "extra" / "extra6_localized_bridge")
    p.add_argument("--n-total-heads", type=int, default=0,
                   help="Universe size for the random-overlap test (0 = infer n_layers*n_heads).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _stage2_dir(step5_root: Path, model_short: str) -> Path:
    return step5_root / model_short / "stage2_ablation"


def _iter_heads(container: Any):
    """`heads` is a list of dicts (filtered_heads.json) or a dict keyed by
    '(layer,head)' -> dict (ablation_scores.json). Yield the head dicts."""
    if isinstance(container, dict):
        yield from container.values()
    elif isinstance(container, list):
        yield from container


def load_stage2_heads(step5_root: Path, model_short: str, group: str) -> tuple[set, dict]:
    """Return ({(layer,head)} passing set, {(layer,head): delta_XL} over all candidates).

    group is 'general' or a language code."""
    base = _stage2_dir(step5_root, model_short) / group
    passing: set = set()
    fh = base / "filtered_heads.json"
    if fh.exists():
        for h in _iter_heads(_load(fh).get("heads", [])):
            passing.add((int(h["layer"]), int(h["head"])))
    deltas: dict = {}
    sc = base / "ablation_scores.json"
    if sc.exists():
        for h in _iter_heads(_load(sc).get("heads", [])):
            key = (int(h["layer"]), int(h["head"]))
            deltas[key] = h.get("mean_delta_XL", h.get("delta_XL"))
    return passing, deltas


def _jaccard(a: set, b: set) -> float:
    u = len(a | b)
    return (len(a & b) / u) if u else float("nan")


def _hypergeom_p(overlap: int, n_a: int, n_b: int, n_total: int) -> float:
    # P(overlap >= observed) if B were a random n_b-subset of n_total, given A of size n_a.
    if n_total <= 0 or n_a == 0 or n_b == 0:
        return float("nan")
    return float(scipy_stats.hypergeom.sf(overlap - 1, n_total, n_a, n_b))


def _spearman(orig: dict, loc: dict) -> tuple[float, int]:
    shared = sorted(set(orig) & set(loc))
    shared = [k for k in shared if orig[k] is not None and loc[k] is not None]
    if len(shared) < 3:
        return float("nan"), len(shared)
    x = [orig[k] for k in shared]
    y = [loc[k] for k in shared]
    return float(scipy_stats.spearmanr(x, y).statistic), len(shared)


# ---------------------------------------------------------------------------
# Confounds
# ---------------------------------------------------------------------------

def _filtered_count(data_root: Path, model_short: str, lang: str, kind: str) -> int:
    path = data_root / model_short / "filtered" / lang / f"{kind}_{model_short}_{lang}.json"
    if not path.exists():
        return -1
    obj = _load(path)
    data = obj["data"] if isinstance(obj, dict) and "data" in obj else obj
    return len(data)


def _coverage(localized_dir: Path, lang: str) -> dict:
    path = localized_dir / lang / f"two_hop_{lang}.json"
    if not path.exists():
        return {}
    meta = _load(path).get("metadata", {}) if isinstance(_load(path), dict) else {}
    cov = meta.get("localization_coverage", {}).get(lang, {})
    return cov


def _infer_n_total(step5_root: Path, models: list[str], langs: list[str]) -> int:
    max_layer = max_head = -1
    for m in models:
        for grp in ["general", *langs]:
            for fn in ["filtered_heads.json", "ablation_scores.json"]:
                p = _stage2_dir(step5_root, m) / grp / fn
                if not p.exists():
                    continue
                for h in _iter_heads(_load(p).get("heads", [])):
                    max_layer = max(max_layer, int(h["layer"]))
                    max_head = max(max_head, int(h["head"]))
    if max_layer < 0 or max_head < 0:
        return 0
    return (max_layer + 1) * (max_head + 1)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_model(args, model: str, n_total: int, logger) -> list[dict]:
    loc = f"{model}{args.loc_suffix}"
    rows: list[dict] = []

    groups = [("general", "general", None)] + [("specific", lg, lg) for lg in args.langs]
    for head_group, group_key, lang in groups:
        o_set, o_delta = load_stage2_heads(args.step5_root, model, group_key)
        l_set, l_delta = load_stage2_heads(args.step5_root, loc, group_key)
        if not o_set and not l_set:
            logger.warning("%s/%s: no Stage-2 output for either run -- skipping.", model, group_key)
            continue
        overlap = len(o_set & l_set)
        rho, n_shared = _spearman(o_delta, l_delta)
        row = {
            "model": model, "head_group": head_group, "lang": lang or "-",
            "n_orig": len(o_set), "n_loc": len(l_set), "overlap": overlap,
            "jaccard": _jaccard(o_set, l_set),
            "expected_overlap_random": (len(o_set) * len(l_set) / n_total) if n_total else float("nan"),
            "hypergeom_p": _hypergeom_p(overlap, len(o_set), len(l_set), n_total),
            "spearman_delta_XL": rho, "n_shared_candidates": n_shared,
        }
        # Confounds (per-language groups only)
        if lang is not None:
            row["orig_correct_n"] = _filtered_count(args.data_root, model, lang, "correct")
            row["loc_correct_n"] = _filtered_count(args.data_root, loc, lang, "correct")
            cov = _coverage(args.localized_processed_dir, lang)
            row["cov_e1"] = cov.get("e1_localized_frac")
            row["cov_e2"] = cov.get("e2_localized_frac")
        rows.append(row)
        logger.info(
            "%s %-8s %-3s  n_orig=%3d n_loc=%3d overlap=%3d  J=%.3f  p=%.1e  rho=%.2f",
            model, head_group, (lang or "-"), len(o_set), len(l_set), overlap,
            row["jaccard"], row["hypergeom_p"], rho,
        )
    return rows


def main() -> None:
    args = parse_args()
    logger = setup_logging("extra6_localized_entity_bridge", LOG_DIR)
    n_total = args.n_total_heads or _infer_n_total(args.step5_root, args.models, args.langs)
    logger.info("Extra 6 -- models=%s  loc_suffix=%s  n_total_heads(universe)=%d",
                args.models, args.loc_suffix, n_total)

    all_rows: list[dict] = []
    for model in args.models:
        if not (_stage2_dir(args.step5_root, f"{model}{args.loc_suffix}")).exists():
            logger.warning("Localized run not found for %s (expected %s%s). Run the localized "
                           "pipeline first -- skipping.", model, model, args.loc_suffix)
            continue
        all_rows += compare_model(args, model, n_total, logger)

    args.output_root.mkdir(parents=True, exist_ok=True)
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(args.output_root / "comparison.csv", index=False)
        with (args.output_root / "comparison.jsonl").open("w", encoding="utf-8") as f:
            for r in all_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # Compact verdict per model x head_group
        verdict = (df.groupby(["model", "head_group"])
                     .agg(mean_jaccard=("jaccard", "mean"),
                          mean_spearman=("spearman_delta_XL", "mean"))
                     .reset_index().to_dict(orient="records"))
        with (args.output_root / "summary.json").open("w", encoding="utf-8") as f:
            json.dump({"n_total_heads": n_total, "rows": len(all_rows), "verdict": verdict},
                      f, ensure_ascii=False, indent=2)
        logger.info("Saved comparison (%d rows) -> %s", len(all_rows), args.output_root)
    else:
        logger.warning("No comparison rows produced. Has the localized pipeline been run?")


if __name__ == "__main__":
    main()
