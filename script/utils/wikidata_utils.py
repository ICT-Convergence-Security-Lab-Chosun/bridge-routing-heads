from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
USER_AGENT = (
    "multilingual-hop/0.1 "
    "(https://www.wikidata.org/wiki/Wikidata:Data_access; research script)"
)


def resolve_cache_file(cache_path: Path) -> Path:
    if cache_path.suffix == ".json":
        return cache_path
    return cache_path / "entity_labels_cache.json"


def load_cache(cache_path: Path) -> dict[str, dict[str, str]]:
    cache_file = resolve_cache_file(cache_path)
    if not cache_file.exists():
        return {}
    with cache_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(cache: dict[str, dict[str, str]], cache_path: Path) -> None:
    cache_file = resolve_cache_file(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _request_labels(
    qcodes: list[str],
    langs: list[str],
    max_retries: int = 5,
) -> dict[str, dict[str, str]]:
    params = {
        "action": "wbgetentities",
        "format": "json",
        "props": "labels",
        "ids": "|".join(qcodes),
        "languages": "|".join(sorted(set(langs + ["en"]))),
    }
    session = requests.Session()
    headers = {"User-Agent": USER_AGENT}

    for attempt in range(max_retries):
        response = session.get(WIKIDATA_API, params=params, headers=headers, timeout=30)
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait_s = float(retry_after) if retry_after else 2 ** attempt
            time.sleep(wait_s)
            continue
        if response.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
        payload = response.json()
        entities = payload.get("entities", {})
        out: dict[str, dict[str, str]] = {}
        for qcode in qcodes:
            labels = entities.get(qcode, {}).get("labels", {})
            en_label = labels.get("en", {}).get("value", qcode)
            out[qcode] = {}
            for lang in langs:
                out[qcode][lang] = labels.get(lang, {}).get("value") or en_label
            out[qcode]["en"] = en_label
            out[qcode]["_updated_at"] = datetime.now().replace(microsecond=0).isoformat()
        return out

    raise RuntimeError(f"Wikidata API failed after {max_retries} retries for batch={qcodes[:3]}...")


def get_entity_labels(
    qcode: str,
    langs: list[str],
    cache: dict[str, dict[str, str]],
) -> dict[str, str]:
    cached = cache.get(qcode, {})
    if all(cached.get(lang) for lang in langs):
        return {lang: cached.get(lang, cached.get("en", qcode)) for lang in langs}

    fetched = _request_labels([qcode], langs)[qcode]
    merged = {**cached, **fetched}
    cache[qcode] = merged
    return {lang: merged.get(lang, merged.get("en", qcode)) for lang in langs}


def batch_get_labels(
    qcodes: list[str],
    langs: list[str],
    cache_path: Path,
    batch_size: int = 50,
    logger: logging.Logger | None = None,
) -> dict[str, dict[str, str]]:
    log = logger or LOGGER
    cache = load_cache(cache_path)
    unique_qcodes = sorted({q for q in qcodes if isinstance(q, str) and q.startswith("Q")})
    missing = [
        qcode
        for qcode in unique_qcodes
        if not all(cache.get(qcode, {}).get(lang) for lang in langs)
    ]

    for start in tqdm(range(0, len(missing), batch_size), desc="Wikidata labels"):
        batch = missing[start : start + batch_size]
        if not batch:
            continue
        try:
            fetched = _request_labels(batch, langs)
        except Exception as exc:
            log.warning(
                "Wikidata batch failed for sample=%s: %s. Using cached EN/QID fallback.",
                batch[:3],
                exc,
            )
            fetched = {
                qcode: {lang: cache.get(qcode, {}).get("en", qcode) for lang in langs}
                for qcode in batch
            }
        for qcode, labels in fetched.items():
            cache[qcode] = {**cache.get(qcode, {}), **labels}
        save_cache(cache, cache_path)

    save_cache(cache, cache_path)
    return {
        qcode: {lang: cache.get(qcode, {}).get(lang, cache.get(qcode, {}).get("en", qcode)) for lang in langs}
        for qcode in unique_qcodes
    }
