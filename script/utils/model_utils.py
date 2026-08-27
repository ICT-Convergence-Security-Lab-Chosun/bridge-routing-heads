from __future__ import annotations

import re
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.common import unique_nonempty

# Models at/above this parameter count (in billions) are sharded across GPUs via
# device_map="auto" instead of being loaded onto a single device. Parsed from the
# model name's size tag (e.g. 70B, 72B, 405B) so it is not tied to one string.
_SHARD_MIN_BILLIONS = 40.0


def _model_size_billions(model_name: str) -> float:
    """Best-effort parameter count (billions) parsed from the model name, else 0."""
    sizes = re.findall(r"(\d+(?:\.\d+)?)\s*b\b", model_name.lower())
    return max((float(s) for s in sizes), default=0.0)


def _should_shard(model_name: str, device: str) -> bool:
    return device == "auto" or _model_size_billions(model_name) >= _SHARD_MIN_BILLIONS


def load_model_and_tokenizer(
    model_name: str,
    device: str,
    torch_dtype: str = "auto",
    hf_token: str | None = None,
    trust_remote_code: bool = False,
    **model_extra_kwargs,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=trust_remote_code,
        clean_up_tokenization_spaces=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "token": hf_token,
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
        **model_extra_kwargs,
    }
    if _should_shard(model_name, device):
        model_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)
    model.eval()
    return model, tokenizer


def _model_input_device(model) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def predict_next_tokens(model, tokenizer, prompt: str, n_tokens: int = 10) -> str:
    return predict_next_tokens_batch(model, tokenizer, [prompt], n_tokens=n_tokens)[0]


def predict_next_tokens_batch(model, tokenizer, prompts: list[str], n_tokens: int = 10) -> list[str]:
    if not prompts:
        return []
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    device = _model_input_device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=n_tokens,
        )
    prompt_len = inputs["input_ids"].shape[-1]
    new_tokens = generated[:, prompt_len:]
    decoded = tokenizer.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [text.strip().replace("\n", " ") for text in decoded]


def _normalize_answer_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def check_answer(prediction: str, answer_labels: Iterable[str]) -> bool:
    pred = _normalize_answer_text(prediction)
    answers = unique_nonempty(answer_labels)
    return any(_normalize_answer_text(answer) in pred for answer in answers)
