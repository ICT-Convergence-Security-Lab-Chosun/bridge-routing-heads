from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from utils.common import build_metadata, load_json, save_json, set_seed, setup_logging
from utils.prompt_utils import build_prompts_for_lang
from utils.wikidata_utils import batch_get_labels, resolve_cache_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build multilingual HoppingTooLate datasets.")
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "data/raw/two_hop.csv")  # Path to the original HoppingTooLate two_hop CSV
    parser.add_argument(
        "--templates",
        type=Path,
        default=PROJECT_ROOT / "script/config/relation_templates.json",
    )  # Path to the per-relation language template JSON
    parser.add_argument("--langs", nargs="+", default=["en", "ko", "zh", "ja", "es", "id", "vi", "hi"])  # List of language codes to generate
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/processed")  # Root directory for per-language JSON output
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data/wikidata_cache")  # Directory to cache Wikidata labels
    parser.add_argument("--max-samples", type=int, default=None)  # Maximum number of samples for debug/partial runs
    parser.add_argument("--sample-strategy", choices=["head", "random"], default="head")  # Whether to take the first N or randomly sample when --max-samples is set
    parser.add_argument("--seed", type=int, default=42)  # Seed for random sampling and reproducibility
    parser.add_argument(
        "--entity-prompt-label",
        choices=["english", "localized"],
        default="english",
        help="Which entity label to place in the prompt text. 'english' (default) keeps "
             "the original English source-entity anchor for e1/e2. 'localized' uses the "
             "target-language Wikidata label (English fallback when none exists) -- used to "
             "test whether general Bridge Heads still emerge without the English anchor.",
    )  # Entity anchor in prompts: english (original) or localized (target-language)
    return parser.parse_args()


def _value(row: pd.Series, *candidates: str) -> Any:
    for candidate in candidates:
        if candidate in row and pd.notna(row[candidate]):
            return row[candidate]
    raise KeyError(f"None of the candidate columns exist: {candidates}")


def _qid(row: pd.Series, key: str) -> str:
    return str(_value(row, f"{key}_qid", key))


def _en_label(row: pd.Series, key: str) -> str:
    return str(_value(row, f"{key}_label_en", f"{key}_label"))


def _relation_name(row: pd.Series, key: str) -> str:
    return str(_value(row, f"{key}_type", f"{key}_name", key))


def _relation_pid(row: pd.Series, key: str) -> str:
    return str(_value(row, f"{key}_pid", key))


def _record_id(row: pd.Series) -> str:
    return "_".join([_qid(row, "e1"), _relation_pid(row, "r1"), _qid(row, "e2"), _relation_pid(row, "r2"), _qid(row, "e3")])


def _hop_id(row: pd.Series) -> str:
    return "_".join([_qid(row, "e1"), _relation_pid(row, "r1"), _qid(row, "e2"), _relation_pid(row, "r2")])


def _label_for(
    label_map: dict[str, dict[str, str]],
    qid: str,
    lang: str,
    en_fallback: str,
) -> str:
    label = label_map.get(qid, {}).get(lang)
    if not label or label == qid:
        return en_fallback
    return label


def _select_rows(df: pd.DataFrame, max_samples: int | None, strategy: str, seed: int) -> pd.DataFrame:
    if max_samples is None or max_samples >= len(df):
        return df.copy()
    if strategy == "random":
        return df.sample(n=max_samples, random_state=seed).sort_index().reset_index(drop=True)
    return df.head(max_samples).copy()


def _relation_column(df: pd.DataFrame, key: str) -> str:
    for candidate in [f"{key}_type", f"{key}_name", key]:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"Cannot find relation column for {key!r}; expected one of {key}_type, {key}_name, {key}")


def _get_relations_from_df(df: pd.DataFrame) -> list[str]:
    r1_col = _relation_column(df, "r1")
    r2_col = _relation_column(df, "r2")
    return sorted(set(df[r1_col].astype(str)) | set(df[r2_col].astype(str)))


def _validate_templates(df: pd.DataFrame, templates: dict[str, Any], langs: list[str]) -> None:
    relations = _get_relations_from_df(df)
    missing = [relation for relation in relations if relation not in templates]
    if missing:
        raise KeyError(f"Missing relation templates: {missing}")
    missing_langs = [
        (relation, lang)
        for relation in relations
        for lang in langs
        if lang not in templates[relation]
    ]
    if missing_langs:
        raise KeyError(f"Missing relation language templates: {missing_langs}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    logger = setup_logging("step1_build_multilingual", LOG_DIR)

    df = pd.read_csv(args.source)
    total_source_rows = len(df)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df = _select_rows(df, args.max_samples, args.sample_strategy, args.seed)
    templates = load_json(args.templates)
    _validate_templates(df, templates, args.langs)

    all_qids = []
    for key in ["e1", "e2", "e3"]:
        all_qids.extend(_qid(row, key) for _, row in df.iterrows())
    non_qids = sorted({qid for qid in all_qids if not qid.startswith("Q")})
    if non_qids:
        logger.warning(
            "Found %d non-QID entity values; Wikidata lookup will skip them and use CSV label fallback. sample=%s",
            len(non_qids),
            non_qids[:10],
        )
    label_map = batch_get_labels(all_qids, args.langs, resolve_cache_file(args.cache_dir), logger=logger)

    localized_prompts = args.entity_prompt_label == "localized"
    # Coverage: per language, how many e1/e2 got a genuine localized label (!= English).
    loc_cov: dict[str, dict[str, int]] = {
        lang: {"e1_localized": 0, "e2_localized": 0, "n": 0} for lang in args.langs
    }

    lang_records: dict[str, list[dict[str, Any]]] = {lang: [] for lang in args.langs}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Build records"):
        e1_qid, e2_qid, e3_qid = _qid(row, "e1"), _qid(row, "e2"), _qid(row, "e3")
        e1_en, e2_en, e3_en = _en_label(row, "e1"), _en_label(row, "e2"), _en_label(row, "e3")
        r1, r2 = _relation_name(row, "r1"), _relation_name(row, "r2")
        r1_pid, r2_pid = _relation_pid(row, "r1"), _relation_pid(row, "r2")

        for lang in args.langs:
            e1_label = _label_for(label_map, e1_qid, lang, e1_en)
            e2_label = _label_for(label_map, e2_qid, lang, e2_en)
            e3_label = _label_for(label_map, e3_qid, lang, e3_en)

            loc_cov[lang]["n"] += 1
            if e1_label != e1_en:
                loc_cov[lang]["e1_localized"] += 1
            if e2_label != e2_en:
                loc_cov[lang]["e2_localized"] += 1

            # Anchor in the prompt: English (original) or localized target-language label.
            e1_prompt = e1_label if localized_prompts else e1_en
            e2_prompt = e2_label if localized_prompts else e2_en
            prompts = build_prompts_for_lang(
                r1=r1,
                r2=r2,
                e1_label=e1_label,
                e2_label=e2_label,
                templates=templates,
                lang=lang,
                e1_prompt_label=e1_prompt,
                e2_prompt_label=e2_prompt,
            )

            lang_records[lang].append(
                {
                    "id": _record_id(row),
                    "hop_id": _hop_id(row),
                    "source_id": int(row["id"]) if "id" in row and pd.notna(row["id"]) else None,
                    "lang": lang,
                    "e1": {"qid": e1_qid, "label": e1_label, "label_en": e1_en},
                    "e2": {"qid": e2_qid, "label": e2_label, "label_en": e2_en},
                    "e3": {"qid": e3_qid, "label": e3_label, "label_en": e3_en},
                    "e1_label_en": e1_en,
                    "e2_label_en": e2_en,
                    "e3_label_en": e3_en,
                    "r1": r1,
                    "r2": r2,
                    "r1_pid": r1_pid,
                    "r2_pid": r2_pid,
                    "prompts": prompts,
                    "t1_search_key": e1_en,
                }
            )

    # Localization coverage report (fraction of e1/e2 with a genuine target-language label).
    coverage = {
        lang: {
            "n": c["n"],
            "e1_localized_frac": (c["e1_localized"] / c["n"]) if c["n"] else 0.0,
            "e2_localized_frac": (c["e2_localized"] / c["n"]) if c["n"] else 0.0,
        }
        for lang, c in loc_cov.items()
    }
    logger.info("entity_prompt_label=%s  localization coverage:", args.entity_prompt_label)
    for lang, c in coverage.items():
        logger.info("  %-3s  e1=%.1f%%  e2=%.1f%%  (n=%d)",
                    lang, 100 * c["e1_localized_frac"], 100 * c["e2_localized_frac"], c["n"])

    metadata = build_metadata(
        "step1_build_multilingual",
        args,
        total_raw=total_source_rows,
        total_written={lang: len(records) for lang, records in lang_records.items()},
        cache_file=str(resolve_cache_file(args.cache_dir)),
        entity_prompt_label=args.entity_prompt_label,
        localization_coverage=coverage,
    )
    for lang, records in lang_records.items():
        out_path = args.output_dir / lang / f"two_hop_{lang}.json"
        save_json(records, out_path, metadata)
        logger.info("Saved %d records for lang=%s -> %s", len(records), lang, out_path)


if __name__ == "__main__":
    main()
