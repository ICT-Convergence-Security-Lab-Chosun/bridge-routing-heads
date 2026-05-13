from __future__ import annotations

import argparse
import json
import logging
import platform
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def today() -> str:
    return datetime.now().strftime("%Y%m%d")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, argparse.Namespace):
        return {k: _jsonable(v) for k, v in vars(value).items()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    return value


def library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name in ["numpy", "pandas", "torch", "transformers", "requests", "tqdm"]:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", None)
        except Exception:
            versions[name] = None
    return versions


def build_metadata(step: str, args: argparse.Namespace, **extra: Any) -> dict[str, Any]:
    metadata = {
        "step": step,
        "generated_at": now_iso(),
        "args": _jsonable(args),
        "library_versions": library_versions(),
    }
    metadata.update(_jsonable(extra))
    return metadata


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "data" in obj:
        return obj["data"]
    return obj


def load_json_document(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path, metadata: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {"metadata": metadata, "data": data} if metadata is not None else data
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def append_jsonl(record: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_done_ids(path: Path) -> set[str]:
    return {str(record["id"]) for record in iter_jsonl(path)}


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def setup_logging(name: str, log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_path = log_dir / f"{name}_{today()}.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Logging to %s", log_path)
    return logger


def find_latest_eval(output_dir: Path, model_short: str, lang: str) -> Path:
    eval_dir = output_dir / model_short / "eval" / lang
    candidates = sorted(
        eval_dir.glob(f"eval_{model_short}_{lang}_*.json"),
        key=lambda p: (p.stat().st_mtime, p.name),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No eval file found under {eval_dir}")
    return candidates[0]


def unique_nonempty(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
