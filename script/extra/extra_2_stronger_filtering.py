"""Extra 2: Stronger Patchscopes Filtering with Configurable Hyperparameters.

Reads  output/step5_BridgeHead_Ablation_Patchscopes/{model_short}/stage3_patchscopes/{lang}/patchscopes_scores.json
for all requested models and languages, applies a stricter threshold than the
original "successes > 0" filter, and writes the results as CSV files.

Filtering metrics
-----------------
  successes     : number of patchscopes runs that generated the bridge entity
  encoding_rate : fraction of runs that succeeded  (= successes / total)
  both          : head must pass BOTH the successes AND the encoding_rate threshold

Thresholds are supplied as CLI arguments so every re-run is fully reproducible.

Output  (output/extra/extra_2_stronger_filtering/)
------
  all_heads.csv   — one row per passing head:  model, lang, layer, head
  summary.csv     — one row per (model, lang):  model, lang, num_heads
  {model}_{lang}_filtered.csv  — per-pair file: layer, head  (convenient for downstream scripts)

Usage examples
--------------
# Filter by successes >= 2 (stricter than the original successes >= 1)
python script/extra/extra_2_stronger_filtering.py \\
    --models llama31_70 qwen25_72 \\
    --langs ko zh ja es \\
    --metric successes \\
    --min-successes 2

# Filter by encoding_rate >= 0.04
python script/extra/extra_2_stronger_filtering.py \\
    --models llama31_70 qwen25_72 \\
    --langs ko zh ja es \\
    --metric encoding_rate \\
    --min-enc-rate 0.04

# Must satisfy BOTH thresholds simultaneously
python script/extra/extra_2_stronger_filtering.py \\
    --models llama31_70 qwen25_72 \\
    --langs ko zh ja es \\
    --metric both \\
    --min-successes 1 \\
    --min-enc-rate 0.02
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATCHSCOPES_ROOT = PROJECT_ROOT / "output" / "step5_BridgeHead_Ablation_Patchscopes"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "extra" / "extra_2_stronger_filtering"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODELS = ["llama31_70", "qwen25_72"]
DEFAULT_LANGS = ["ko", "zh", "ja", "es"]
DEFAULT_METRIC: Literal["successes", "encoding_rate", "both"] = "successes"
DEFAULT_MIN_SUCCESSES = 2
DEFAULT_MIN_ENC_RATE = 0.04


# ---------------------------------------------------------------------------
# Core filtering
# ---------------------------------------------------------------------------

def load_patchscopes_scores(model: str, lang: str) -> dict | None:
    """Load patchscopes_scores.json for the given model/lang pair.

    Returns None if the file does not exist.
    """
    path = PATCHSCOPES_ROOT / model / "stage3_patchscopes" / lang / "patchscopes_scores.json"
    if not path.exists():
        print(f"  [WARN] Not found: {path}")
        return None
    with path.open() as f:
        return json.load(f)


def filter_heads(
    data: dict,
    metric: Literal["successes", "encoding_rate", "both"],
    min_successes: int,
    min_enc_rate: float,
) -> list[dict]:
    """Return list of head dicts that pass the specified filtering criterion.

    Each returned dict contains: layer (int), head (int), successes (int),
    encoding_rate (float), total (int).
    """
    passing = []
    for head_data in data["heads"].values():
        s = head_data["successes"]
        r = head_data["encoding_rate"]

        if metric == "successes":
            ok = s >= min_successes
        elif metric == "encoding_rate":
            ok = r >= min_enc_rate
        elif metric == "both":
            ok = s >= min_successes and r >= min_enc_rate
        else:
            raise ValueError(f"Unknown metric: {metric!r}")

        if ok:
            passing.append(
                {
                    "layer": head_data["layer"],
                    "head": head_data["head"],
                    "successes": s,
                    "encoding_rate": r,
                    "total": head_data["total"],
                }
            )

    # Sort by layer then head for deterministic output
    passing.sort(key=lambda x: (x["layer"], x["head"]))
    return passing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extra 2: stronger patchscopes filtering with configurable thresholds."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        metavar="MODEL",
        help=f"Model short-names to process (default: {DEFAULT_MODELS}).",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=DEFAULT_LANGS,
        metavar="LANG",
        help=f"Language codes to process (default: {DEFAULT_LANGS}).",
    )
    parser.add_argument(
        "--metric",
        choices=["successes", "encoding_rate", "both"],
        default=DEFAULT_METRIC,
        help=(
            "Filtering metric: "
            "'successes' = head.successes >= --min-successes; "
            "'encoding_rate' = head.encoding_rate >= --min-enc-rate; "
            "'both' = must satisfy both conditions simultaneously. "
            f"(default: {DEFAULT_METRIC})"
        ),
    )
    parser.add_argument(
        "--min-successes",
        type=int,
        default=DEFAULT_MIN_SUCCESSES,
        metavar="N",
        help=(
            "Minimum number of successful patchscopes probes required "
            f"(used when --metric is 'successes' or 'both'; default: {DEFAULT_MIN_SUCCESSES})."
        ),
    )
    parser.add_argument(
        "--min-enc-rate",
        type=float,
        default=DEFAULT_MIN_ENC_RATE,
        metavar="RATE",
        help=(
            "Minimum encoding rate [0, 1] required "
            f"(used when --metric is 'encoding_rate' or 'both'; default: {DEFAULT_MIN_ENC_RATE})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Override output directory (default: output/extra/extra_2_stronger_filtering/).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir: Path = args.output_dir if args.output_dir is not None else OUTPUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Print configuration
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Extra 2 — Stronger Patchscopes Filtering")
    print("=" * 60)
    print(f"  models          : {args.models}")
    print(f"  langs           : {args.langs}")
    print(f"  metric          : {args.metric}")
    if args.metric in ("successes", "both"):
        print(f"  min_successes   : {args.min_successes}")
    if args.metric in ("encoding_rate", "both"):
        print(f"  min_enc_rate    : {args.min_enc_rate}")
    print(f"  output dir      : {out_dir}")
    print()

    all_rows: list[dict] = []       # for all_heads.csv
    summary_rows: list[dict] = []   # for summary.csv

    # ------------------------------------------------------------------
    # Process each (model, lang) pair
    # ------------------------------------------------------------------
    for model in args.models:
        for lang in args.langs:
            print(f"[{model}/{lang}]", end="  ")

            data = load_patchscopes_scores(model, lang)
            if data is None:
                print("skipped (file not found)")
                continue

            n_candidates = len(data["heads"])
            passing = filter_heads(
                data,
                metric=args.metric,
                min_successes=args.min_successes,
                min_enc_rate=args.min_enc_rate,
            )
            n_passing = len(passing)

            print(
                f"candidates={n_candidates}  passing={n_passing}  "
                f"(original filter: {data.get('n_candidates', '?')} → kept with successes>0)"
            )

            # --- per-pair CSV (only layer + head) ---
            pair_rows = [{"layer": h["layer"], "head": h["head"]} for h in passing]
            pair_df = pd.DataFrame(pair_rows, columns=["layer", "head"])
            pair_csv = out_dir / f"{model}_{lang}_filtered.csv"
            pair_df.to_csv(pair_csv, index=False)

            # --- accumulate for combined outputs ---
            for h in passing:
                all_rows.append(
                    {
                        "model": model,
                        "lang": lang,
                        "layer": h["layer"],
                        "head": h["head"],
                    }
                )
            summary_rows.append(
                {
                    "model": model,
                    "lang": lang,
                    "num_heads": n_passing,
                }
            )

    # ------------------------------------------------------------------
    # Write combined CSVs
    # ------------------------------------------------------------------
    all_df = pd.DataFrame(all_rows, columns=["model", "lang", "layer", "head"])
    all_csv = out_dir / "all_heads.csv"
    all_df.to_csv(all_csv, index=False)

    summary_df = pd.DataFrame(summary_rows, columns=["model", "lang", "num_heads"])
    summary_csv = out_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print()
    print(f"Total passing heads : {summary_df['num_heads'].sum()}")
    print()
    print("Output files:")
    print(f"  {all_csv}")
    print(f"  {summary_csv}")
    for row in summary_rows:
        fname = out_dir / f"{row['model']}_{row['lang']}_filtered.csv"
        print(f"  {fname}")


if __name__ == "__main__":
    main()
