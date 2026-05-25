from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Any

from utils.common import (
    build_metadata,
    find_latest_eval,
    load_json,
    load_json_document,
    save_json,
    set_seed,
    setup_logging,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"


EVAL_KEYS = [
    "two_hop_correct",
    "two_hop_pred",
    "first_hop_correct",
    "first_hop_pred",
    "second_hop_correct",
    "second_hop_pred",
    "shortcut1_correct",
    "shortcut1_pred",
    "shortcut2_correct",
    "shortcut2_pred",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter multilingual HoppingTooLate eval results.")
    parser.add_argument("--model-short", required=True)  # Short model name used in Step 2
    parser.add_argument("--langs", nargs="+", default=["en", "ko", "zh", "ja", "es"])  # List of language codes to filter/cross-classify
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/processed")  # Root directory for Step 1 output data
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data")  # Root directory for Step 2 eval and Step 3 filtered results
    parser.add_argument("--seed", type=int, default=42)  # Seed for reproducibility metadata recording
    return parser.parse_args()


def is_valid_eval(ev: dict[str, Any]) -> bool:
    return bool(
        ev.get("first_hop_correct")
        and ev.get("second_hop_correct")
        and not ev.get("shortcut1_correct")
        and not ev.get("shortcut2_correct")
    )


def slim_eval(ev: dict[str, Any]) -> dict[str, Any]:
    return {key: ev.get(key) for key in EVAL_KEYS}


def merge_record(record: dict[str, Any], ev: dict[str, Any]) -> dict[str, Any]:
    merged = dict(record)
    merged["model"] = ev.get("model")
    merged["eval"] = slim_eval(ev)
    return merged


def build_cross_record(
    rid: str,
    data_by_lang: dict[str, dict[str, dict[str, Any]]],
    eval_by_lang: dict[str, dict[str, dict[str, Any]]],
    langs: list[str],
    model_short: str,
) -> dict[str, Any]:
    first = data_by_lang[langs[0]][rid]
    return {
        "id": rid,
        "hop_id": first.get("hop_id"),
        "source_id": first.get("source_id"),
        "model": model_short,
        "r1": first.get("r1"),
        "r2": first.get("r2"),
        "r1_pid": first.get("r1_pid"),
        "r2_pid": first.get("r2_pid"),
        "langs": {
            lang: {
                "e1_label": data_by_lang[lang][rid]["e1"]["label"],
                "e2_label": data_by_lang[lang][rid]["e2"]["label"],
                "e3_label": data_by_lang[lang][rid]["e3"]["label"],
                "e1_label_en": data_by_lang[lang][rid].get("e1_label_en"),
                "e2_label_en": data_by_lang[lang][rid].get("e2_label_en"),
                "e3_label_en": data_by_lang[lang][rid].get("e3_label_en"),
                "prompts": data_by_lang[lang][rid]["prompts"],
                "t1_search_key": data_by_lang[lang][rid].get("t1_search_key"),
                "eval": slim_eval(eval_by_lang[lang][rid]),
            }
            for lang in langs
        },
    }


def load_step1_counts(data_dir: Path, langs: list[str]) -> tuple[int | None, dict[str, int]]:
    if not langs:
        return None, {}
    doc_path = data_dir / langs[0] / f"two_hop_{langs[0]}.json"
    doc = load_json_document(doc_path)
    metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
    total_raw = metadata.get("total_raw")
    total_written = metadata.get("total_written", {})
    return total_raw, total_written if isinstance(total_written, dict) else {}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    logger = setup_logging(f"step3_filter_multilingual_{args.model_short}", LOG_DIR)

    data_by_lang: dict[str, dict[str, dict[str, Any]]] = {}
    eval_by_lang: dict[str, dict[str, dict[str, Any]]] = {}
    total_raw, per_lang_raw = load_step1_counts(args.data_dir, args.langs)

    for lang in args.langs:
        records = load_json(args.data_dir / lang / f"two_hop_{lang}.json")
        eval_path = find_latest_eval(args.output_dir, args.model_short, lang)
        eval_records = load_json(eval_path)
        data_by_lang[lang] = {record["id"]: record for record in records}
        eval_by_lang[lang] = {record["id"]: record for record in eval_records}
        logger.info("Loaded lang=%s data=%d eval=%d from %s", lang, len(records), len(eval_records), eval_path)

    valid_ids: dict[str, set[str]] = {}
    correct_ids: dict[str, set[str]] = {}
    incorrect_ids: dict[str, set[str]] = {}
    per_lang_counts: dict[str, dict[str, int]] = {}

    metadata = build_metadata("step3_filter_multilingual", args, model_short=args.model_short)
    for lang in args.langs:
        valid, correct, incorrect = [], [], []
        for rid, record in data_by_lang[lang].items():
            ev = eval_by_lang[lang].get(rid)
            if ev is None or not is_valid_eval(ev):
                continue
            merged = merge_record(record, ev)
            valid.append(merged)
            if ev.get("two_hop_correct"):
                correct.append(merged)
            else:
                incorrect.append(merged)

        base = args.output_dir / args.model_short / "filtered" / lang
        save_json(valid, base / f"valid_{args.model_short}_{lang}.json", metadata)
        save_json(correct, base / f"correct_{args.model_short}_{lang}.json", metadata)
        save_json(incorrect, base / f"incorrect_{args.model_short}_{lang}.json", metadata)

        valid_ids[lang] = {record["id"] for record in valid}
        correct_ids[lang] = {record["id"] for record in correct}
        incorrect_ids[lang] = {record["id"] for record in incorrect}
        per_lang_counts[lang] = {
            "valid": len(valid),
            "correct": len(correct),
            "incorrect": len(incorrect),
        }
        logger.info("Filtered lang=%s valid=%d correct=%d incorrect=%d", lang, len(valid), len(correct), len(incorrect))

    cross_base = args.output_dir / args.model_short / "filtered" / "cross_lang"
    cross_counts: dict[str, int] = {}

    common_valid = set.intersection(*(valid_ids[lang] for lang in args.langs)) if args.langs else set()
    all_correct = [
        build_cross_record(rid, data_by_lang, eval_by_lang, args.langs, args.model_short)
        for rid in sorted(common_valid)
        if all(rid in correct_ids[lang] for lang in args.langs)
    ]
    save_json(all_correct, cross_base / f"all_correct_{args.model_short}.json", metadata)
    cross_counts["all_correct"] = len(all_correct)

    for lang1, lang2 in combinations(args.langs, 2):
        pair_valid = valid_ids[lang1] & valid_ids[lang2]
        pair_correct = [
            build_cross_record(rid, data_by_lang, eval_by_lang, [lang1, lang2], args.model_short)
            for rid in sorted(pair_valid)
            if rid in correct_ids[lang1] and rid in correct_ids[lang2]
        ]
        key = f"{lang1}_{lang2}_both_correct"
        save_json(pair_correct, cross_base / f"{key}_{args.model_short}.json", metadata)
        cross_counts[key] = len(pair_correct)
        logger.info("Cross pair=%s-%s both_correct=%d", lang1, lang2, len(pair_correct))

    if "en" in args.langs:
        for target_lang in [lang for lang in args.langs if lang != "en"]:
            candidates = valid_ids["en"] & valid_ids[target_lang]
            cross = [
                build_cross_record(rid, data_by_lang, eval_by_lang, ["en", target_lang], args.model_short)
                for rid in sorted(candidates)
                if rid in correct_ids["en"] and rid in incorrect_ids[target_lang]
            ]
            key = f"en_correct_{target_lang}_incorrect"
            save_json(cross, cross_base / f"{key}_{args.model_short}.json", metadata)
            cross_counts[key] = len(cross)

        if "ko" in args.langs and "zh" in args.langs:
            candidates = valid_ids["en"] & valid_ids["ko"] & valid_ids["zh"]
            cross = [
                build_cross_record(rid, data_by_lang, eval_by_lang, ["en", "ko", "zh"], args.model_short)
                for rid in sorted(candidates)
                if rid in correct_ids["en"] and rid in incorrect_ids["ko"] and rid in incorrect_ids["zh"]
            ]
            key = "en_correct_ko_incorrect_zh_incorrect"
            save_json(cross, cross_base / f"{key}_{args.model_short}.json", metadata)
            cross_counts[key] = len(cross)

    summary = {
        "model": args.model_short,
        "generated_at": metadata["generated_at"],
        "total_raw": total_raw
        if total_raw is not None
        else (len(next(iter(data_by_lang.values()))) if data_by_lang else 0),
        "per_lang_raw": per_lang_raw,
        "per_lang": per_lang_counts,
        "cross_lang": cross_counts,
        "metadata": metadata,
    }
    summary_path = args.output_dir / args.model_short / "filtered" / f"split_summary_{args.model_short}.json"
    save_json(summary, summary_path)
    logger.info("Saved summary -> %s", summary_path)


if __name__ == "__main__":
    main()
