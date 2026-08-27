"""Extra 6 prep: restrict the localized dataset to originally-correct cases.

Why
---
Re-evaluating all 82k localized prompts per language does not fit in the review
window. We therefore keep only the records that were CORRECT in the original
(English-anchor) run of the given model: cases the model failed *with* the
English anchor are not going to be solved by localizing the entity names, and
the head-identification pipeline (step4 BRS, step5 Stage-2 ablation) only ever
consumes correct samples anyway -- so this restriction shrinks compute ~30x
without biasing which heads can be identified.

Reads
-----
  data/processed_localized/{lang}/two_hop_{lang}.json      (step1 --entity-prompt-label localized)
  data/{model_short}/filtered/{lang}/correct_{model_short}_{lang}.json   (original run)

Writes
------
  data/processed_localized_{model_short}/{lang}/two_hop_{lang}.json
      -- same schema as step1 output, restricted to originally-correct ids,
         with localization coverage recomputed on the subset.

Then run the localized pipeline with:
  --data-dir data/processed_localized_{model_short}  --model-short {model_short}_loc

Usage
-----
python script/extra/extra_6_prepare_localized_subset.py --model-short llama31_70 --langs en ko zh ja es
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"

import sys
sys.path.insert(0, str(PROJECT_ROOT / "script"))

from utils.common import load_json, load_json_document, save_json, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Restrict localized data to originally-correct cases (Extra 6 prep).")
    p.add_argument("--model-short", required=True,
                   help="ORIGINAL model_short whose correct sets define the subset (e.g. llama31_70).")
    p.add_argument("--langs", nargs="+", default=["en", "ko", "zh", "ja", "es"])
    p.add_argument("--localized-dir", type=Path,
                   default=PROJECT_ROOT / "data" / "processed_localized")
    p.add_argument("--filtered-root", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Default: data/processed_localized_{model_short}.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = args.output_dir or (PROJECT_ROOT / "data" / f"processed_localized_{args.model_short}")
    logger = setup_logging(f"extra6_prepare_localized_subset_{args.model_short}", LOG_DIR)

    for lang in args.langs:
        loc_path = args.localized_dir / lang / f"two_hop_{lang}.json"
        doc = load_json_document(loc_path)
        records = doc["data"] if isinstance(doc, dict) and "data" in doc else doc
        src_meta = doc.get("metadata", {}) if isinstance(doc, dict) else {}

        correct_path = (args.filtered_root / args.model_short / "filtered" / lang
                        / f"correct_{args.model_short}_{lang}.json")
        correct_ids = {r["id"] for r in load_json(correct_path)}

        subset = [r for r in records if r["id"] in correct_ids]

        # Recompute localization coverage on the subset.
        n = len(subset)
        e1_loc = sum(1 for r in subset if r["e1"]["label"] != r["e1_label_en"])
        e2_loc = sum(1 for r in subset if r["e2"]["label"] != r["e2_label_en"])
        coverage = {
            "n": n,
            "e1_localized_frac": (e1_loc / n) if n else 0.0,
            "e2_localized_frac": (e2_loc / n) if n else 0.0,
        }

        metadata = {
            "step": "extra6_prepare_localized_subset",
            "restricted_to": f"originally-correct ids of {args.model_short} ({correct_path.name})",
            "n_localized_full": len(records),
            "n_correct_ids": len(correct_ids),
            "n_subset": n,
            "entity_prompt_label": src_meta.get("args", {}).get("entity_prompt_label", "localized"),
            "localization_coverage": {lang: coverage},
            "source_step1_metadata": {k: src_meta.get(k) for k in ("step", "generated_at", "args")},
        }
        out_path = out_root / lang / f"two_hop_{lang}.json"
        save_json(subset, out_path, metadata)
        logger.info(
            "%s: %d/%d kept (correct ids=%d)  cov e1=%.1f%% e2=%.1f%%  -> %s",
            lang, n, len(records), len(correct_ids),
            100 * coverage["e1_localized_frac"], 100 * coverage["e2_localized_frac"], out_path,
        )


if __name__ == "__main__":
    main()
