from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from utils.common import (
    append_jsonl,
    build_metadata,
    load_done_ids,
    load_json,
    load_jsonl_records,
    now_iso,
    save_json,
    set_seed,
    setup_logging,
    unique_nonempty,
)
from utils.model_utils import check_answer, load_model_and_tokenizer, predict_next_tokens_batch
from utils.prompt_utils import wrap_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multilingual HoppingTooLate prompts.")
    parser.add_argument("--model", required=True)  # HuggingFace model id 또는 로컬 모델 경로 ex) meta-llama/Llama-3.1-70B, Qwen/Qwen2.5-72B
    parser.add_argument("--model-short", required=True)  # 출력 폴더/파일명에 사용할 짧은 모델 이름
    parser.add_argument("--langs", nargs="+", default=["en", "ko", "zh", "ja", "es"])  # 평가할 언어 코드 목록
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/processed")  # Step 1 출력 데이터 루트
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data")  # eval/checkpoint 결과 저장 루트
    parser.add_argument("--n-tokens", type=int, default=10)  # 각 prompt 뒤에 greedy decoding으로 생성할 토큰 수
    parser.add_argument("--batch-size", type=int, default=512)  # 한 번에 평가할 record 수; 70B는 4~16부터 시도 권장
    parser.add_argument("--gpu-index", type=int, default=0)  # 단일 GPU 실행 시 사용할 CUDA index
    parser.add_argument("--device", default=None)  # 강제 실행 device; auto면 HF device_map='auto' 사용
    parser.add_argument("--torch-dtype", default="auto")  # 모델 로드 dtype; auto/bfloat16/float16 등
    parser.add_argument("--hf-token", default=None)  # gated/private 모델 접근용 HuggingFace token
    parser.add_argument("--trust-remote-code", action="store_true")  # custom model code가 필요한 HF 모델에서 활성화
    parser.add_argument("--checkpoint-every", type=int, default=1)  # checkpoint 저장 주기 표기용; 현재는 샘플마다 즉시 저장
    parser.add_argument("--resume", action="store_true")  # 기존 checkpoint.jsonl에서 완료 id를 읽어 이어서 평가
    parser.add_argument(
        "--max-samples",
        "--max-sample",
        dest="max_samples",
        type=int,
        default=None,
        help="언어별로 이번 실행에서 평가할 최대 샘플 수. 0이면 제한 없이 실행합니다. --resume 사용 시 checkpoint에 없는 pending 샘플에서 이어서 자릅니다.",
    )
    parser.add_argument("--seed", type=int, default=42)  # torch/random/numpy 재현성 제어용 seed
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 0:
        parser.error("--max-samples must be >= 0")
    return args


def _device_from_args(args: argparse.Namespace) -> str:
    if args.device:
        return args.device
    if torch.cuda.is_available():
        return f"cuda:{args.gpu_index}"
    return "cpu"


def _answer_labels(record: dict[str, Any], entity_key: str) -> list[str]:
    entity = record.get(entity_key, {})
    return unique_nonempty(
        [
            entity.get("label"),
            entity.get("label_en"),
            record.get(f"{entity_key}_label_en"),
        ]
    )


EVAL_SPECS = [
    ("two_hop", "two_hop", "e3", "two_hop"),
    ("first_hop", "first_hop", "e2", "first_hop"),
    ("second_hop", "second_hop", "e3", "second_hop"),
    ("shortcut1", "shortcut_no_e1", "e3", "shortcut1"),
    ("shortcut2", "shortcut_no_r1", "e3", "shortcut2"),
]


def evaluate_records_batch(
    model,
    tokenizer,
    records: list[dict[str, Any]],
    n_tokens: int,
    model_short: str,
) -> list[dict[str, Any]]:
    results = [
        {
            "id": record["id"],
            "hop_id": record.get("hop_id"),
            "source_id": record.get("source_id"),
            "lang": record["lang"],
            "model": model_short,
        }
        for record in records
    ]

    _printed_sample = True # 평가 프롬프트 샘플 출력 제어용 플래그
    for _, prompt_key, answer_key, out_prefix in EVAL_SPECS:
        prompts = [
            wrap_prompt(record["prompts"][prompt_key], record["lang"])
            for record in records
        ]
        if not _printed_sample:
            print(f"\n{'='*60}\n[PROMPT SAMPLE] key={prompt_key!r}\n{'='*60}")
            print(prompts[0])
            print("="*60)
            _printed_sample = True
        predictions = predict_next_tokens_batch(model, tokenizer, prompts, n_tokens=n_tokens)
        for result, record, prediction in zip(results, records, predictions):
            answers = _answer_labels(record, answer_key)
            # Trim at first newline: entity names are single-line; everything
            # after the first \n is drift (follow-up sentences, MCQs, etc.)
            trimmed = prediction.split("\n")[0].strip()
            result[f"{out_prefix}_correct"] = check_answer(trimmed, answers)
            result[f"{out_prefix}_pred"] = trimmed

    return results


def iter_batches(records: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = _device_from_args(args)
    logger = setup_logging(f"step2_evaluate_multilingual_{args.model_short}", LOG_DIR)
    logger.info("Loading model=%s on device=%s", args.model, device)
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        device=device,
        torch_dtype=args.torch_dtype,
        hf_token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
    )

    for lang in args.langs:
        data_path = args.data_dir / lang / f"two_hop_{lang}.json"
        records = load_json(data_path)
        eval_dir = args.output_dir / args.model_short / "eval" / lang
        ckpt_path = eval_dir / "checkpoint.jsonl"
        if not args.resume and ckpt_path.exists():
            ckpt_path.unlink()
        done_ids = load_done_ids(ckpt_path) if args.resume else set()
        logger.info("Evaluating lang=%s records=%d resume_done=%d", lang, len(records), len(done_ids))

        all_pending_records = [record for record in records if record["id"] not in done_ids]
        pending_records = (
            all_pending_records[: args.max_samples]
            if args.max_samples is not None and args.max_samples > 0
            else all_pending_records
        )
        logger.info(
            "Run limit lang=%s pending_total=%d run_records=%d max_samples=%s",
            lang,
            len(all_pending_records),
            len(pending_records),
            args.max_samples,
        )
        progress = tqdm(total=len(pending_records), desc=f"eval [{lang}]")
        for batch in iter_batches(pending_records, max(1, args.batch_size)):
            eval_results = evaluate_records_batch(model, tokenizer, batch, args.n_tokens, args.model_short)
            for eval_result in eval_results:
                append_jsonl(eval_result, ckpt_path)
            progress.update(len(batch))
        progress.close()

        all_results = load_jsonl_records(ckpt_path)
        timestamp = now_iso().replace("-", "").replace(":", "")
        out_path = eval_dir / f"eval_{args.model_short}_{lang}_{timestamp}.json"
        metadata = build_metadata(
            "step2_evaluate_multilingual",
            args,
            model=args.model,
            model_short=args.model_short,
            lang=lang,
            checkpoint=str(ckpt_path),
            total_records=len(records),
            total_evaluated=len(all_results),
            evaluated_this_run=len(pending_records),
            pending_before_run=len(all_pending_records),
            pending_after_run=max(0, len(all_pending_records) - len(pending_records)),
            max_samples=args.max_samples,
        )
        save_json(all_results, out_path, metadata)
        logger.info("Saved eval results for lang=%s -> %s", lang, out_path)


if __name__ == "__main__":
    main()
