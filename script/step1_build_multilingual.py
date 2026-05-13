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
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "data/raw/two_hop.csv")  # 원본 HoppingTooLate two_hop CSV 경로
    parser.add_argument(
        "--templates",
        type=Path,
        default=PROJECT_ROOT / "script/config/relation_templates.json",
    )  # relation별 언어 템플릿 JSON 경로
    parser.add_argument("--langs", nargs="+", default=["en", "ko", "zh", "ja", "es", "id", "vi", "hi"])  # 생성할 언어 코드 목록
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/processed")  # 언어별 JSON 저장 루트
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data/wikidata_cache")  # Wikidata label 캐시 저장 위치
    parser.add_argument("--max-samples", type=int, default=None)  # 디버그/부분 실험용 최대 샘플 수
    parser.add_argument("--sample-strategy", choices=["head", "random"], default="head")  # max-samples 적용 시 앞에서 자를지 무작위 샘플링할지 선택
    parser.add_argument("--seed", type=int, default=42)  # 무작위 샘플링 및 재현성 제어용 seed
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
            prompts = build_prompts_for_lang(
                r1=r1,
                r2=r2,
                e1_label=e1_label,
                e2_label=e2_label,
                templates=templates,
                lang=lang,
                e1_prompt_label=e1_en,
                e2_prompt_label=e2_en,
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

    metadata = build_metadata(
        "step1_build_multilingual",
        args,
        total_raw=total_source_rows,
        total_written={lang: len(records) for lang, records in lang_records.items()},
        cache_file=str(resolve_cache_file(args.cache_dir)),
    )
    for lang, records in lang_records.items():
        out_path = args.output_dir / lang / f"two_hop_{lang}.json"
        save_json(records, out_path, metadata)
        logger.info("Saved %d records for lang=%s -> %s", len(records), lang, out_path)


if __name__ == "__main__":
    main()
