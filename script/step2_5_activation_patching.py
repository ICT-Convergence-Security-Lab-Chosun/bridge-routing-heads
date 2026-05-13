"""Step 4: Layer-wise cross-lingual activation patching.

Ports the old stage2 analysis into multilingual_hop without importing the
backup project or step2 script. Step 3 filtered files choose the correct
samples; multilingual_hop/data/processed provides the actual prompt records.
"""

from __future__ import annotations

import argparse
import json
import logging
from itertools import permutations
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from utils.common import load_json, set_seed, setup_logging

# ---------------------------------------------------------------------------
# Step 4 — specific constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_LANGS = ["en", "ko", "zh", "ja", "es"]
PROMPT_KEYS = ["two_hop", "first_hop", "second_hop", "shortcut_no_e1", "shortcut_no_r1"]

PATCH_TYPES = ["attn", "mlp", "residual"]
HOOK_TEMPLATES = {
    "attn": "blocks.{layer}.hook_attn_out",
    "mlp": "blocks.{layer}.hook_mlp_out",
    "residual": "blocks.{layer}.hook_resid_post",
}

INSTRUCTIONS: dict[str, str] = {
    "en": "Answer briefly with only the answer.",
    "ko": "정답만 짧게 답하시오.",
    "zh": "只简短回答答案。",
    "ja": "答えだけを短く答えなさい。",
    "es": "Responde brevemente solo con la respuesta.",
}


# ---------------------------------------------------------------------------
# Prompt construction — local copy of Step 2's prompt format
# ---------------------------------------------------------------------------
def wrap_prompt(question: str, lang: str) -> str:
    instruction = INSTRUCTIONS.get(lang, INSTRUCTIONS["en"])
    return f"Instruction:{instruction} Question:{question} Answer:"


def build_prompt(record: dict[str, Any], prompt_key: str) -> str:
    return wrap_prompt(record["prompts"][prompt_key], record["lang"])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def parse_lang_pair(pair_str: str) -> tuple[str, str]:
    parts = pair_str.split("-", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid lang pair format: {pair_str!r}; expected langA-langB")
    return parts[0], parts[1]


def build_lang_pairs(langs: list[str], lang_pairs: list[str] | None) -> list[tuple[str, str]]:
    if lang_pairs:
        return [parse_lang_pair(pair_str) for pair_str in lang_pairs]
    return list(permutations(langs, 2))


def load_lang_records(data_dir: Path, lang: str) -> dict[str, dict[str, Any]]:
    records = load_json(data_dir / lang / f"two_hop_{lang}.json")
    return {record["id"]: record for record in records}


def _cross_dir_candidates(output_dir: Path, model_short: str, langs: list[str]) -> list[Path]:
    filtered_dir = output_dir / model_short / "filtered"
    candidates: list[Path] = []
    if set(langs) == set(DEFAULT_LANGS):
        candidates.append(filtered_dir / "cross_lang_all_lang")
    candidates.append(filtered_dir / f"cross_lang_{'_'.join(langs)}")
    candidates.append(filtered_dir / "cross_lang_all_lang")

    if filtered_dir.exists():
        for path in sorted(filtered_dir.glob("cross_lang_*")):
            suffix = path.name.removeprefix("cross_lang_")
            if suffix == "all_lang":
                matches = set(langs) == set(DEFAULT_LANGS)
            else:
                matches = set(suffix.split("_")) == set(langs)
            if matches:
                candidates.append(path)

    candidates.append(filtered_dir / "cross_lang")

    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def resolve_pair_file(
    output_dir: Path,
    model_short: str,
    langs: list[str],
    lang_a: str,
    lang_b: str,
) -> Path:
    filenames = [
        f"{lang_a}_{lang_b}_both_correct_{model_short}.json",
        f"{lang_b}_{lang_a}_both_correct_{model_short}.json",
        f"all_correct_{model_short}.json",
    ]
    checked: list[Path] = []
    for cross_dir in _cross_dir_candidates(output_dir, model_short, langs):
        for filename in filenames:
            candidate = cross_dir / filename
            checked.append(candidate)
            if candidate.exists():
                return candidate
    checked_text = "\n  ".join(str(path) for path in checked)
    raise FileNotFoundError(
        f"No filtered pair file found for {lang_a}->{lang_b}. Checked:\n  {checked_text}"
    )


def load_pairwise_samples(
    data_dir: Path,
    output_dir: Path,
    model_short: str,
    langs: list[str],
    lang_a: str,
    lang_b: str,
    max_samples: int | None,
) -> tuple[list[dict[str, Any]], Path]:
    pair_file = resolve_pair_file(output_dir, model_short, langs, lang_a, lang_b)
    filtered_records = load_json(pair_file)
    target_ids = [record["id"] for record in filtered_records]
    if max_samples is not None and max_samples > 0:
        target_ids = target_ids[:max_samples]

    data_a = load_lang_records(data_dir, lang_a)
    data_b = load_lang_records(data_dir, lang_b)
    samples = []
    for rid in target_ids:
        if rid not in data_a or rid not in data_b:
            continue
        samples.append({"id": rid, "record_a": data_a[rid], "record_b": data_b[rid]})
    return samples, pair_file


# ---------------------------------------------------------------------------
# TransformerLens model loading
# ---------------------------------------------------------------------------
def _torch_dtype_from_arg(value: str) -> torch.dtype:
    aliases = {
        "auto": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return aliases[value.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported --torch-dtype: {value}") from exc


def load_hooked_model(
    model_name: str,
    device: str,
    torch_dtype: str,
    hf_token: str | None,
    trust_remote_code: bool,
):
    from transformer_lens import HookedTransformer

    kwargs: dict[str, Any] = {"dtype": _torch_dtype_from_arg(torch_dtype)}
    if hf_token:
        kwargs["token"] = hf_token
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    model = HookedTransformer.from_pretrained_no_processing(model_name, device=device, **kwargs)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Position detection, metrics, logit lens, and patching
# ---------------------------------------------------------------------------
def _text_char_to_token(model, tokens: torch.Tensor) -> tuple[str, list[int]]:
    token_strs = [model.tokenizer.decode([tid]) for tid in tokens[0].tolist()]
    text = ""
    char_to_tok: list[int] = []
    for i, token_str in enumerate(token_strs):
        for _ in token_str:
            char_to_tok.append(i)
        text += token_str
    return text, char_to_tok


def find_answer_boundary(model, tokens: torch.Tensor) -> dict[str, int]:
    text, char_to_tok = _text_char_to_token(model, tokens)
    idx = text.rfind("Answer:")
    if idx >= 0 and idx < len(char_to_tok):
        answer_tok_start = char_to_tok[idx]
        colon_char_idx = idx + len("Answer")
        colon_tok = char_to_tok[colon_char_idx] if colon_char_idx < len(char_to_tok) else tokens.shape[1] - 1
        pre_answer_tok = max(0, answer_tok_start - 1)
    else:
        pre_answer_tok = max(0, tokens.shape[1] - 2)
        colon_tok = tokens.shape[1] - 1
    return {"pre_answer": pre_answer_tok, "answer_colon": colon_tok}


def find_question_start(model, tokens: torch.Tensor) -> int:
    text, char_to_tok = _text_char_to_token(model, tokens)
    idx = text.rfind("Question:")
    if idx >= 0 and idx < len(char_to_tok):
        return char_to_tok[idx]
    return 0


@torch.no_grad()
def logit_lens_from_cache(
    model,
    cache,
    n_layers: int,
    seq_len: int,
    top_k: int = 5,
    layer_start: int = 0,
    tok_start: int = 0,
):
    result = {}
    for layer in range(layer_start, n_layers):
        residual = cache[f"blocks.{layer}.hook_resid_post"]
        residual_f32 = residual.float()[:, tok_start:, :]
        normed = model.ln_final(residual_f32)
        logits = model.unembed(normed)
        probs = torch.softmax(logits[0], dim=-1)

        layer_result = {}
        for t_rel, t_abs in enumerate(range(tok_start, seq_len)):
            topk_probs, topk_ids = probs[t_rel].topk(top_k)
            preds = []
            for prob_val, tok_id in zip(topk_probs.tolist(), topk_ids.tolist()):
                token_str = model.tokenizer.decode([tok_id])
                preds.append((token_str, round(prob_val, 6)))
            layer_result[t_abs] = preds
        result[layer] = layer_result
    return result


def compute_logit_diff(clean_logits_vec: torch.Tensor, patched_logits_vec: torch.Tensor) -> float:
    clean_top1_id = clean_logits_vec.argmax().item()
    return (patched_logits_vec[clean_top1_id] - clean_logits_vec[clean_top1_id]).item()


@torch.no_grad()
def compute_kl_at_last_token(
    clean_logits_last: torch.Tensor,
    patched_logits_last: torch.Tensor,
) -> float:
    c = clean_logits_last.float()
    p = patched_logits_last.float()
    clean_log_probs = torch.log_softmax(c, dim=-1)
    patched_log_probs = torch.log_softmax(p, dim=-1)
    clean_probs = torch.softmax(c, dim=-1)
    kl = (clean_probs * (clean_log_probs - patched_log_probs)).sum()
    return max(0.0, kl.item())


@torch.no_grad()
def patch_single_layer(
    model,
    clean_tokens: torch.Tensor,
    source_cache: dict,
    patch_type: str,
    layer: int,
    patch_pos: int,
    source_pos: int,
) -> torch.Tensor:
    hook_name = HOOK_TEMPLATES[patch_type].format(layer=layer)
    cached_act = source_cache[hook_name]

    def hook_fn(value, hook):
        value[:, patch_pos] = cached_act[:, source_pos]
        return value

    model.add_hook(hook_name, hook_fn)
    try:
        logits = model(clean_tokens)
    finally:
        model.reset_hooks()
    return logits[0, -1]


@torch.no_grad()
def patch_single_layer_with_cache(
    model,
    clean_tokens: torch.Tensor,
    source_cache: dict,
    patch_type: str,
    layer: int,
    patch_pos: int,
    source_pos: int,
) -> tuple[torch.Tensor, Any]:
    hook_name = HOOK_TEMPLATES[patch_type].format(layer=layer)
    cached_act = source_cache[hook_name]

    def hook_fn(value, hook):
        value[:, patch_pos] = cached_act[:, source_pos]
        return value

    names_filter = lambda name: (
        name.endswith(".hook_resid_post") and int(name.split(".")[1]) >= layer
    )
    model.add_hook(hook_name, hook_fn)
    try:
        logits, cache = model.run_with_cache(clean_tokens, names_filter=names_filter)
    finally:
        model.reset_hooks()
    return logits[0, -1], cache


# ---------------------------------------------------------------------------
# Per-sample analysis  (Bug 5 fixed: all under torch.no_grad)
# ---------------------------------------------------------------------------
@torch.no_grad()
def analyze_sample(
    model,
    sample: dict,
    prompt_key: str,
    patch_type: str,
    n_layers: int,
    logger: logging.Logger,
) -> dict | None:
    """
    Per-layer patching analysis for one sample.

    For each layer L (independently):
      - Patch layer L at pre_answer position → measure KL & logit diff
      - Patch layer L at answer_colon position → measure KL & logit diff
    Also compute clean logit lens for visualization.
    """
    record_a = sample["record_a"]  # source language
    record_b = sample["record_b"]  # target (clean) language

    raw_question_a = record_a["prompts"][prompt_key]
    raw_question_b = record_b["prompts"][prompt_key]
    prompt_a = build_prompt(record_a, prompt_key)
    prompt_b = build_prompt(record_b, prompt_key)

    tokens_a = model.to_tokens(prompt_a)
    tokens_b = model.to_tokens(prompt_b)

    seq_len_b = tokens_b.shape[1]
    token_strs_b = [model.tokenizer.decode([t]) for t in tokens_b[0].tolist()]

    # Find Answer boundary positions in BOTH languages
    boundary_b = find_answer_boundary(model, tokens_b)
    boundary_a = find_answer_boundary(model, tokens_a)

    # Cache names: need attn/mlp/resid for patching + resid for logit lens
    names_filter = lambda name: name.endswith(
        (".hook_resid_post", ".hook_attn_out", ".hook_mlp_out")
    )

    # Source run (language A)
    _, source_cache = model.run_with_cache(tokens_a, names_filter=names_filter)

    # Clean run (language B)
    clean_logits, clean_cache = model.run_with_cache(tokens_b, names_filter=names_filter)
    clean_logits_last = clean_logits[0, -1].float()  # (vocab,)

    # Clean logit lens (for visualization)
    clean_lens = logit_lens_from_cache(model, clean_cache, n_layers, seq_len_b)

    # Clean logit diff is 0 by definition (clean vs clean = no change)
    clean_logit_diff = 0.0

    question_start_tok = find_question_start(model, tokens_b)

    # Per-layer patching at two key positions
    POSITIONS = {
        "pre_answer": (boundary_b["pre_answer"], boundary_a["pre_answer"]),
        "answer_colon": (boundary_b["answer_colon"], boundary_a["answer_colon"]),
    }

    per_layer_kl: dict[str, list[float]] = {k: [] for k in POSITIONS}
    per_layer_logit_diff: dict[str, list[float]] = {k: [] for k in POSITIONS}

    # patched_logit_lens[pos_name][layer] = {token_idx: [(token_str, prob), ...]}
    # Only store for the question section (tokens from question_start_tok to seq_len)
    # to keep payload size manageable.
    patched_logit_lens: dict[str, dict[int, dict]] = {k: {} for k in POSITIONS}

    for layer in range(n_layers):
        for pos_name, (tgt_pos, src_pos) in POSITIONS.items():
            patched_logits_last, patched_cache = patch_single_layer_with_cache(
                model, tokens_b, source_cache, patch_type,
                layer=layer, patch_pos=tgt_pos, source_pos=src_pos,
            )
            patched_logits_last = patched_logits_last.float()

            kl_val = compute_kl_at_last_token(clean_logits_last, patched_logits_last)
            ld_val = compute_logit_diff(clean_logits_last, patched_logits_last)

            per_layer_kl[pos_name].append(kl_val)
            per_layer_logit_diff[pos_name].append(ld_val)

            # Only compute lens from 'layer' onward; only question tokens (not context)
            pl = logit_lens_from_cache(
                model, patched_cache, n_layers, seq_len_b,
                layer_start=layer,
                tok_start=question_start_tok,
            )
            patched_logit_lens[pos_name][layer] = pl
            del patched_cache

    # Cleanup
    del source_cache, clean_cache
    torch.cuda.empty_cache()

    return {
        "id": sample["id"],
        "tokens": token_strs_b,
        "seq_len": seq_len_b,
        "n_layers": n_layers,
        "question_start_tok": question_start_tok,
        "clean_logit_lens": clean_lens,
        "clean_logit_diff": clean_logit_diff,
        "per_layer_kl": per_layer_kl,
        "per_layer_logit_diff": per_layer_logit_diff,
        "patched_logit_lens": patched_logit_lens,
        "patch_positions": {k: v[0] for k, v in POSITIONS.items()},
        "question_a": raw_question_a,
        "question_b": raw_question_b,
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "final_answer": record_b["e3"]["label"],
        "final_answer_en": record_b["e3"].get("label_en"),
    }


# ---------------------------------------------------------------------------
# HTML generation  — publication-quality, patched logit lens interactive tab
# ---------------------------------------------------------------------------
def build_html(payload: dict, title: str) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
:root {{
  --bg: #fafafa; --surface: #ffffff; --border: #e2e5ea;
  --text: #1c2333; --muted: #6b7585;
  --blue: #2563eb; --orange: #ea580c; --gray-line: #9ca3af;
  --radius: 8px;
  --font: 'IBM Plex Sans','Helvetica Neue',Arial,sans-serif;
  --mono: 'IBM Plex Mono','Fira Code',monospace;
}}
html {{ font-size: 14px; }}
body {{ font-family: var(--font); background: var(--bg); color: var(--text); line-height: 1.5; }}
.page {{ max-width: 1120px; margin: 0 auto; padding: 28px 20px 64px; }}
/* header */
.hdr {{ border-bottom: 1.5px solid var(--border); padding-bottom: 18px; margin-bottom: 24px; }}
.hdr-title {{ font-size: 1.1rem; font-weight: 600; letter-spacing: -.01em; margin-bottom: 8px; }}
.hdr-meta {{ display: flex; flex-wrap: wrap; gap: 8px 22px; font-size: .76rem; color: var(--muted); font-family: var(--mono); }}
.hdr-meta b {{ color: var(--text); font-family: var(--font); font-weight: 500; }}
.badge {{ display:inline-block; padding:2px 8px; border-radius:4px; font-size:.7rem; font-weight:600;
          letter-spacing:.05em; text-transform:uppercase; background:#dbeafe; color:#1e40af; margin-left:8px; }}
/* cards */
.card {{ background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
         margin-bottom: 20px; overflow: hidden; }}
.card-head {{ display:flex; align-items:baseline; gap:10px; padding:12px 18px 10px;
              border-bottom: 1px solid var(--border); }}
.card-head h2 {{ font-size:.86rem; font-weight:600; flex:1; }}
.card-head .sub {{ font-size:.73rem; color:var(--muted); }}
.card-body {{ padding: 18px; }}
/* charts */
.chart-svg-wrap {{ width:100%; overflow:visible; }}
.chart-svg-wrap svg {{ width:100%; height:auto; display:block; overflow:visible; }}
.legend {{ display:flex; flex-wrap:wrap; gap:6px 18px; padding:10px 18px 13px;
           border-top:1px solid var(--border); }}
.legend-item {{ display:flex; align-items:center; gap:6px; font-size:.73rem; color:var(--muted); }}
.legend-sw {{ width:22px; height:2.5px; border-radius:2px; flex-shrink:0; }}
.legend-sw.dashed {{
  background: repeating-linear-gradient(90deg,var(--gray-line) 0 5px,transparent 5px 9px); }}
/* tabs */
.tabs {{ display:flex; gap:2px; flex-wrap:wrap; }}
.tab-btn {{ padding:6px 14px; font-size:.76rem; font-weight:500; cursor:pointer;
            border:1px solid var(--border); border-bottom:none; border-radius:6px 6px 0 0;
            background:#f1f3f7; color:var(--muted); transition:background .1s,color .1s; }}
.tab-btn:hover {{ background:#e8ebf0; color:var(--text); }}
.tab-btn.active {{ background:var(--surface); color:var(--text); font-weight:600; }}
.tab-panel {{ display:none; }}
.tab-panel.active {{ display:block; }}
/* logit lens canvas */
.lens-outer {{ overflow:auto; background:#fff; }}
canvas.lens {{ display:block; image-rendering:pixelated; cursor:crosshair; }}
/* controls */
.ctrl-row {{ display:flex; align-items:center; gap:14px; flex-wrap:wrap;
             padding:10px 18px; border-bottom:1px solid var(--border);
             font-size:.75rem; color:var(--muted); }}
.ctrl-row label {{ display:flex; align-items:center; gap:5px; }}
.ctrl-row select {{ font-size:.76rem; padding:3px 6px; border:1px solid var(--border);
                    border-radius:4px; background:#fff; color:var(--text); cursor:pointer; }}
.ctrl-row input[type=range] {{ width:100px; accent-color:var(--blue); }}
/* tbar */
.tbar-wrap {{ overflow-x:auto; padding:8px 18px 6px; border-bottom:1px solid var(--border); background:#fafafa; }}
.tbar {{ display:flex; gap:2px; width:max-content; }}
.tok {{ padding:2px 4px; border-radius:3px; font-size:.62rem; font-family:var(--mono);
        background:#f1f3f7; border:1px solid var(--border); white-space:nowrap; cursor:default; }}
.tok.hl-a {{ background:#dbeafe; border-color:#3b82f6; font-weight:700; }}
.tok.hl-b {{ background:#fed7aa; border-color:#f97316; font-weight:700; }}
/* detail panel */
.detail-strip {{ display:flex; gap:14px; padding:12px 18px 16px; align-items:flex-start; }}
.detail-panel {{ flex:0 0 300px; border:1px solid var(--border); border-radius:var(--radius);
                 padding:12px; font-size:.76rem; background:#f8fafc;
                 max-height:220px; overflow-y:auto; }}
.detail-panel h3 {{ font-size:.8rem; font-weight:600; margin-bottom:8px; }}
.dtable {{ width:100%; border-collapse:collapse; margin-top:6px; }}
.dtable th,.dtable td {{ padding:3px 7px; text-align:left; font-size:.72rem; }}
.dtable th {{ color:var(--muted); font-weight:500; border-bottom:1px solid var(--border); }}
.dtable td {{ border-bottom:1px solid #f0f2f5; font-family:var(--mono); }}
.colorkey {{ font-size:.72rem; color:var(--muted); line-height:2; }}
.swatch {{ display:inline-block; width:10px; height:10px; border-radius:2px; vertical-align:middle; }}
</style>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
</head>
<body>
<div class="page">

<!-- header -->
<div class="hdr">
  <div class="hdr-title" id="hdrTitle"></div>
  <div class="hdr-meta" id="metaRow"></div>
</div>

<!-- KL Divergence -->
<div class="card">
  <div class="card-head">
    <h2>KL Divergence per Layer</h2>
    <span class="sub">KL(clean ∥ patched) at final output — single-layer patching</span>
  </div>
  <div class="card-body"><div class="chart-svg-wrap" id="klWrap"></div></div>
  <div class="legend" id="klLegend"></div>
</div>

<!-- Logit Diff -->
<div class="card">
  <div class="card-head">
    <h2>Logit Difference per Layer</h2>
    <span class="sub">logit(correct) − max logit(others)</span>
  </div>
  <div class="card-body"><div class="chart-svg-wrap" id="ldWrap"></div></div>
  <div class="legend" id="ldLegend"></div>
</div>

<!-- Logit Lens card -->
<div class="card">
  <div class="card-head">
    <h2>Logit Lens</h2>
    <span class="sub">top-1 predicted token at each (layer, position)</span>
  </div>

  <div style="padding:10px 18px 0;border-bottom:1px solid var(--border)">
    <div class="tabs" id="lensTabs">
      <button class="tab-btn active" data-panel="lp-clean">Clean (no patching)</button>
      <button class="tab-btn" data-panel="lp-patched">Patched logit lens</button>
    </div>
  </div>

  <!-- ── Clean panel ── -->
  <div class="tab-panel active" id="lp-clean">
    <div class="ctrl-row">
      <span>Question section only (context excluded)</span>
      <label>Cell width <input type="range" id="cw-clean" min="18" max="80" value="38">
        <span id="cw-clean-val">38</span>px</label>
    </div>
    <div class="tbar-wrap"><div class="tbar" id="tbarClean"></div></div>
    <div class="lens-outer"><canvas class="lens" id="cvClean"></canvas></div>
    <div class="detail-strip">
      <div class="detail-panel" id="detailClean">
        <h3>Cell detail</h3>
        <p style="color:var(--muted)">Hover a cell.</p>
      </div>
      <div class="colorkey">
        <span class="swatch" style="background:#c8e6c9"></span> low confidence &nbsp;
        <span class="swatch" style="background:#1b5e20"></span> high confidence<br>
        <span class="swatch" style="background:#dbeafe"></span>
        <b style="color:#1d4ed8">pre_answer</b> &nbsp;
        <span class="swatch" style="background:#fed7aa"></span>
        <b style="color:#c2410c">answer_colon</b> — patch positions
      </div>
    </div>
  </div>

  <!-- ── Patched panel ── -->
  <div class="tab-panel" id="lp-patched">
    <div class="ctrl-row">
      <label>Patch position
        <select id="selPos">
          <option value="pre_answer">pre_answer</option>
          <option value="answer_colon">answer_colon</option>
        </select>
      </label>
      <label>Patched layer
        <select id="selLayer"></select>
      </label>
      <label>Cell width <input type="range" id="cw-patch" min="18" max="80" value="38">
        <span id="cw-patch-val">38</span>px</label>
      <span id="patchedKL" style="font-family:var(--mono);color:var(--blue)"></span>
      <span id="patchedLD" style="font-family:var(--mono);color:var(--orange)"></span>
    </div>
    <div class="tbar-wrap"><div class="tbar" id="tbarPatched"></div></div>
    <div class="lens-outer"><canvas class="lens" id="cvPatched"></canvas></div>
    <div class="detail-strip">
      <div class="detail-panel" id="detailPatched">
        <h3>Cell detail</h3>
        <p style="color:var(--muted)">Hover a cell.</p>
      </div>
      <div class="colorkey">
        <span class="swatch" style="background:#c8e6c9"></span> low conf. &nbsp;
        <span class="swatch" style="background:#1b5e20"></span> high conf.<br>
        Cell shows <b>Δ</b> vs clean: <span class="swatch" style="background:#fff3e0"></span>
        top-1 changed &nbsp; <span class="swatch" style="background:#c8e6c9"></span> unchanged
      </div>
    </div>
  </div>
</div><!-- /card -->

</div><!-- /page -->

<script>
const D  = {data_json};
const PP = D.patch_positions || {{}};

// ── utils ─────────────────────────────────────────────────────────────────
function esc(s){{ return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }}
function vis(t){{
  let s=String(t||'');
  s=s.replace(/\\n/g,'↵').replace(/\\t/g,'→').replace(/ /g,'·');
  return s||'∅';
}}

// ── header ────────────────────────────────────────────────────────────────
document.getElementById('hdrTitle').innerHTML =
  'Stage 2 · Per-Layer Activation Patching' +
  `<span class="badge">${{esc(D.patch_type||'')}}</span>`;
const meta = [
  ['ID', D.id||'average'], ['Model', D.model||''],
  ['Lang', (D.lang_a||'')+' → '+(D.lang_b||'')],
  ['Layers', D.n_layers], ['Tokens', D.seq_len],
  ['pre_answer pos', PP.pre_answer], ['answer_colon pos', PP.answer_colon],
];
if(D.final_answer) meta.push(['Answer', D.final_answer]);
const metaRow = document.getElementById('metaRow');
meta.forEach(([k,v])=>{{
  if(v===''||v===undefined||v===null) return;
  const el=document.createElement('span');
  el.innerHTML=`<b>${{esc(String(k))}}</b> ${{esc(String(v))}}`;
  metaRow.appendChild(el);
}});
['question_b','question_a'].forEach(key=>{{
  if(!D[key]) return;
  const el=document.createElement('span');
  el.style.flexBasis='100%';
  el.innerHTML=`<b>${{key==='question_b'?'Q (target)':'Q (source)'}}</b> ${{esc(D[key])}}`;
  metaRow.appendChild(el);
}});

// ── SVG chart builder ─────────────────────────────────────────────────────
function makeSVGChart({{datasets,labels,yLabel='',width=900,height=300,
    padTop=24,padBottom=50,padLeft=70,padRight=20}}){{
  const ns='http://www.w3.org/2000/svg';
  const svg=document.createElementNS(ns,'svg');
  svg.setAttribute('viewBox',`0 0 ${{width}} ${{height}}`);
  svg.setAttribute('xmlns',ns);
  svg.setAttribute('font-family',"'IBM Plex Sans','Helvetica Neue',Arial,sans-serif");
  svg.setAttribute('font-size','11');
  const pW=width-padLeft-padRight, pH=height-padTop-padBottom;
  const n=labels.length;
  let mn=Infinity,mx=-Infinity;
  datasets.forEach(ds=>ds.data.forEach(v=>{{if(v<mn)mn=v;if(v>mx)mx=v;}}));
  if(mn===mx)mx=mn+1;
  const span=mx-mn; mn-=span*.07; mx+=span*.07;
  const xOf=i=>padLeft+(i+.5)*pW/n;
  const yOf=v=>padTop+pH*(1-(v-mn)/(mx-mn));
  function el(tag,attrs,parent){{
    const e=document.createElementNS(ns,tag);
    Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));
    if(parent)parent.appendChild(e); return e;
  }}
  function txt(content,attrs,parent){{
    const e=el('text',attrs,parent); e.textContent=content; return e;
  }}
  el('rect',{{x:0,y:0,width,height,fill:'#ffffff'}},svg);
  const nG=5;
  for(let i=0;i<=nG;i++){{
    const v=mn+(mx-mn)*i/nG, y=yOf(v);
    el('line',{{x1:padLeft,y1:y,x2:width-padRight,y2:y,stroke:'#e5e7eb','stroke-width':'1'}},svg);
    const lbl=Math.abs(v)<.001&&Math.abs(v)>0?v.toExponential(1):v.toFixed(Math.abs(v)<.01?4:Math.abs(v)<10?3:1);
    txt(lbl,{{x:padLeft-6,y:y+4,'text-anchor':'end',fill:'#6b7585'}},svg);
  }}
  const step=n<=32?1:Math.ceil(n/32);
  labels.forEach((lbl,i)=>{{
    if(i%step!==0)return;
    txt(lbl,{{x:xOf(i),y:padTop+pH+14,'text-anchor':'middle',fill:'#6b7585'}},svg);
  }});
  txt('Layer',{{x:padLeft+pW/2,y:height-4,'text-anchor':'middle',fill:'#6b7585'}},svg);
  if(yLabel){{
    const yL=el('text',{{x:12,y:padTop+pH/2,'text-anchor':'middle',fill:'#6b7585',
      transform:`rotate(-90,12,${{padTop+pH/2}})`}},svg);
    yL.textContent=yLabel;
  }}
  el('rect',{{x:padLeft,y:padTop,width:pW,height:pH,fill:'none',stroke:'#d1d5db','stroke-width':'1'}},svg);
  datasets.forEach(ds=>{{
    const pts=ds.data.map((v,i)=>`${{xOf(i).toFixed(2)}},${{yOf(v).toFixed(2)}}`).join(' ');
    const poly=el('polyline',{{points:pts,fill:'none',stroke:ds.color,
      'stroke-width':ds.dash?'1.5':'2','stroke-linejoin':'round','stroke-linecap':'round'}},svg);
    if(ds.dash)poly.setAttribute('stroke-dasharray',ds.dash);
    if(!ds.dash)ds.data.forEach((v,i)=>el('circle',{{cx:xOf(i).toFixed(2),cy:yOf(v).toFixed(2),
      r:'3.5',fill:ds.color,stroke:'#fff','stroke-width':'1'}},svg));
  }});
  return svg;
}}

function buildChart(containerId, legendId, datasets, yLabel){{
  const labels=Array.from({{length:D.n_layers}},(_,i)=>String(i));
  document.getElementById(containerId).appendChild(makeSVGChart({{datasets,labels,yLabel}}));
  const leg=document.getElementById(legendId);
  datasets.forEach(ds=>{{
    leg.innerHTML+=`<div class="legend-item">
      <span class="legend-sw${{ds.dash?' dashed':''}}" style="${{ds.dash?'':'background:'+ds.color}}"></span>
      ${{esc(ds.label)}}</div>`;
  }});
}}

// KL
if(D.per_layer_kl){{
  const ds=[];
  if(D.per_layer_kl.pre_answer)    ds.push({{label:'pre_answer',   data:D.per_layer_kl.pre_answer,   color:'#2563eb'}});
  if(D.per_layer_kl.answer_colon)  ds.push({{label:'answer_colon', data:D.per_layer_kl.answer_colon, color:'#ea580c'}});
  buildChart('klWrap','klLegend',ds,'KL divergence');
}}
// Logit diff
if(D.per_layer_logit_diff){{
  const ds=[];
  if(D.clean_logit_diff!==undefined)
    ds.push({{label:'clean baseline',data:Array(D.n_layers).fill(D.clean_logit_diff),color:'#9ca3af',dash:'6 4'}});
  if(D.per_layer_logit_diff.pre_answer)
    ds.push({{label:'patched @ pre_answer',data:D.per_layer_logit_diff.pre_answer,color:'#2563eb'}});
  if(D.per_layer_logit_diff.answer_colon)
    ds.push({{label:'patched @ answer_colon',data:D.per_layer_logit_diff.answer_colon,color:'#ea580c'}});
  buildChart('ldWrap','ldLegend',ds,'logit diff');
}}

// ── Logit Lens renderer ───────────────────────────────────────────────────
// question section only: from question_start_tok to seq_len-1
const Q_START = D.question_start_tok || 0;
const Q_TOKS  = Array.from({{length: D.seq_len - Q_START}}, (_,i) => Q_START + i);

const ROW_HDR = 32;   // px for layer number column
const CELL_H  = 20;   // row height px

function probColor(p){{
  const t=Math.min(Math.max(p,0),1);
  return `rgb(${{Math.round(241+t*(27-241))}},${{Math.round(248+t*(94-248))}},${{Math.round(233+t*(32-233))}})`;
}}

// ---- build token bar
function buildTbar(elId, tokIdxArr){{
  const el=document.getElementById(elId);
  el.innerHTML='';
  tokIdxArr.forEach(i=>{{
    const span=document.createElement('span');
    span.className='tok'+(i===PP.pre_answer?' hl-a':i===PP.answer_colon?' hl-b':'');
    span.textContent=vis(D.tokens[i]||'');
    span.title=`pos ${{i}}`;
    el.appendChild(span);
  }});
}}

// ---- render a lens canvas
// lensData: {{layerIdx: {{tokIdx: [[token,prob],...]}}}}  (string keys)
// mode: 'clean' | 'diff'  (diff = highlight cells where top-1 changed vs clean)
function renderLens(canvas, tokIdxArr, lensData, cellW, mode){{
  const nL=D.n_layers, nT=tokIdxArr.length;
  const W=ROW_HDR+nT*cellW, H=nL*CELL_H;
  canvas.width=W; canvas.height=H;
  canvas.style.width=W+'px'; canvas.style.height=H+'px';
  const ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,W,H);

  const cleanLens = D.clean_logit_lens || {{}};

  for(let li=0;li<nL;li++){{
    // layer label
    ctx.fillStyle='#6b7585';
    ctx.font=`9px "IBM Plex Mono",monospace`;
    ctx.textAlign='right'; ctx.textBaseline='middle';
    ctx.fillText(String(li), ROW_HDR-3, li*CELL_H+CELL_H/2);

    // patched lens omits layers < curLayer; fall back to clean lens for those rows
    const layerData = (lensData[String(li)] !== undefined)
        ? lensData[String(li)]
        : (D.clean_logit_lens[String(li)] || {{}});
    const cleanLayer= (cleanLens[String(li)]||{{}});

    for(let ti=0;ti<nT;ti++){{
      const tokIdx=tokIdxArr[ti];
      const preds=(layerData[String(tokIdx)])||[];
      const top=preds[0];
      const prob=top?top[1]:0;
      const topToken=top?top[0]:'';
      const x=ROW_HDR+ti*cellW, y=li*CELL_H;

      // background
      let bg;
      if(mode==='diff'){{
        const cleanPreds=(cleanLayer[String(tokIdx)])||[];
        const cleanTop=cleanPreds[0]?cleanPreds[0][0]:'';
        bg = (topToken!==''&&topToken!==cleanTop) ? `rgba(255,243,224,${{0.5+prob*0.5}})` : probColor(prob);
      }} else {{
        if(tokIdx===PP.pre_answer)   bg=`rgba(219,234,254,${{0.25+prob*0.55}})`;
        else if(tokIdx===PP.answer_colon) bg=`rgba(254,215,170,${{0.25+prob*0.55}})`;
        else bg=probColor(prob);
      }}
      ctx.fillStyle=bg;
      ctx.fillRect(x,y,cellW-1,CELL_H-1);

      // top-1 token label inside cell
      if(top && cellW>=18){{
        const label=vis(topToken);
        const fontSize=Math.min(9, Math.max(7, Math.floor(cellW/4.5)));
        ctx.font=`${{fontSize}}px "IBM Plex Mono",monospace`;
        ctx.textAlign='center'; ctx.textBaseline='middle';
        // text color: dark on light bg, keep readable
        const brightness = (mode==='diff'&&topToken!==''&&topToken!==(((cleanLens[String(li)]||{{}})[String(tokIdx)])||[['']])[0][0])
          ? 60 : Math.round(50+prob*30);
        ctx.fillStyle=`rgb(${{brightness}},${{brightness}},${{brightness}})`;
        // clip long labels
        const maxChars=Math.floor((cellW-4)/fontSize*1.8);
        const shown=label.length>maxChars?label.slice(0,maxChars)+'…':label;
        ctx.fillText(shown, x+cellW/2, y+CELL_H/2);
      }}

      // separator
      ctx.fillStyle='rgba(200,200,200,0.2)';
      ctx.fillRect(x+cellW-1,y,1,CELL_H);
    }}
    ctx.fillStyle='rgba(180,180,180,0.2)';
    ctx.fillRect(0,li*CELL_H+CELL_H-1,W,1);
    // blue boundary line marks where patching influence begins
    if(mode==='diff' && li===curLayer){{
      ctx.save();
      ctx.strokeStyle='#2563eb';
      ctx.lineWidth=2;
      ctx.beginPath();
      ctx.moveTo(0,li*CELL_H);
      ctx.lineTo(W,li*CELL_H);
      ctx.stroke();
      ctx.restore();
    }}
  }}
}}

// ---- hover → detail panel
function attachHover(canvas, tokIdxArr, lensData, detailId){{
  canvas.addEventListener('mousemove', e=>{{
    const r=canvas.getBoundingClientRect();
    const sx=canvas.width/r.width, sy=canvas.height/r.height;
    const cx=(e.clientX-r.left)*sx, cy=(e.clientY-r.top)*sy;
    const ti=Math.floor((cx-ROW_HDR)/((canvas.width-ROW_HDR)/tokIdxArr.length));
    const li=Math.floor(cy/CELL_H);
    if(ti<0||ti>=tokIdxArr.length||li<0||li>=D.n_layers) return;
    const tokIdx=tokIdxArr[ti];
    const preds=((lensData[String(li)]||{{}})[String(tokIdx)])||[];
    const cleanPreds=((D.clean_logit_lens[String(li)]||{{}})[String(tokIdx)])||[];
    const panel=document.getElementById(detailId);
    let html=`<h3>Layer ${{li}}, Token ${{tokIdx}}</h3>
      <div style="color:var(--muted);margin-bottom:6px">
        <b style="color:var(--text)">${{esc(vis(D.tokens[tokIdx]||''))}}</b></div>
      <table class="dtable"><tr><th>#</th><th>Token</th><th>Prob</th></tr>`;
    preds.forEach(([t,p],i)=>{{
      const changed=cleanPreds[i]&&cleanPreds[i][0]!==t;
      html+=`<tr style="${{changed?'color:#ea580c':''}}">
        <td>${{i+1}}</td><td>${{esc(vis(t))}}</td><td>${{p.toFixed(4)}}</td></tr>`;
    }});
    html+='</table>';
    if(cleanPreds.length){{
      html+=`<div style="margin-top:8px;font-size:.7rem;color:var(--muted)">Clean top-1: <b>${{esc(vis(cleanPreds[0][0]))}}</b></div>`;
    }}
    panel.innerHTML=html;
  }});
}}

// ── Build clean tab ───────────────────────────────────────────────────────
let cwClean=38;
function rebuildClean(){{
  buildTbar('tbarClean', Q_TOKS);
  renderLens(document.getElementById('cvClean'), Q_TOKS, D.clean_logit_lens||{{}}, cwClean, 'clean');
  attachHover(document.getElementById('cvClean'), Q_TOKS, D.clean_logit_lens||{{}}, 'detailClean');
}}
document.getElementById('cw-clean').addEventListener('input', e=>{{
  cwClean=parseInt(e.target.value);
  document.getElementById('cw-clean-val').textContent=cwClean;
  rebuildClean();
}});

// ── Build patched tab ─────────────────────────────────────────────────────
let cwPatch=38, curPos='pre_answer', curLayer=0;

// populate layer selector
const selLayer=document.getElementById('selLayer');
for(let i=0;i<D.n_layers;i++){{
  const opt=document.createElement('option');
  opt.value=i; opt.textContent='Layer '+i;
  selLayer.appendChild(opt);
}}

function updatePatchInfo(){{
  const kl=(D.per_layer_kl||{{}})[curPos];
  const ld=(D.per_layer_logit_diff||{{}})[curPos];
  if(kl) document.getElementById('patchedKL').textContent=`KL=${{kl[curLayer].toFixed(4)}}`;
  if(ld) document.getElementById('patchedLD').textContent=`LD=${{ld[curLayer].toFixed(4)}}`;
}}

function rebuildPatched(){{
  buildTbar('tbarPatched', Q_TOKS);
  const patchedLens=((D.patched_logit_lens||{{}})[curPos]||{{}})[String(curLayer)]||{{}};
  renderLens(document.getElementById('cvPatched'), Q_TOKS, patchedLens, cwPatch, 'diff');
  attachHover(document.getElementById('cvPatched'), Q_TOKS, patchedLens, 'detailPatched');
  updatePatchInfo();
}}

document.getElementById('selPos').addEventListener('change', e=>{{curPos=e.target.value; rebuildPatched();}});
selLayer.addEventListener('change', e=>{{curLayer=parseInt(e.target.value); rebuildPatched();}});
document.getElementById('cw-patch').addEventListener('input', e=>{{
  cwPatch=parseInt(e.target.value);
  document.getElementById('cw-patch-val').textContent=cwPatch;
  rebuildPatched();
}});

// ── Tab switching ─────────────────────────────────────────────────────────
document.getElementById('lensTabs').addEventListener('click', e=>{{
  const btn=e.target.closest('.tab-btn');
  if(!btn) return;
  document.querySelectorAll('#lensTabs .tab-btn').forEach(b=>b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById(btn.dataset.panel).classList.add('active');
}});

// ── Init ──────────────────────────────────────────────────────────────────
rebuildClean();
rebuildPatched();
</script>
</body>
</html>"""


def write_html(payload: dict, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    html = build_html(payload, title)
    path.write_text(html, encoding="utf-8")
    json_path = path.with_suffix(".json")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Average aggregation  (updated for new data structure)
# ---------------------------------------------------------------------------
def aggregate_results(all_results: list[dict]) -> dict:
    if not all_results:
        return {}

    n_layers = all_results[0]["n_layers"]
    n = len(all_results)
    pos_names = list(all_results[0]["per_layer_kl"].keys())

    # Average per-layer KL and logit diff
    avg_kl: dict[str, list[float]] = {k: [0.0] * n_layers for k in pos_names}
    avg_ld: dict[str, list[float]] = {k: [0.0] * n_layers for k in pos_names}
    avg_clean_ld = 0.0

    for r in all_results:
        avg_clean_ld += r["clean_logit_diff"]
        for pos in pos_names:
            for l in range(n_layers):
                avg_kl[pos][l] += r["per_layer_kl"][pos][l]
                avg_ld[pos][l] += r["per_layer_logit_diff"][pos][l]

    avg_clean_ld /= n
    for pos in pos_names:
        for l in range(n_layers):
            avg_kl[pos][l] /= n
            avg_ld[pos][l] /= n

    return {
        "n_layers": n_layers,
        "seq_len": 0,
        "tokens": [],
        "clean_logit_lens": {},
        "clean_logit_diff": avg_clean_ld,
        "per_layer_kl": avg_kl,
        "per_layer_logit_diff": avg_ld,
        "patch_positions": all_results[0].get("patch_positions", {}),
        "num_samples": n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 4: multilingual layer-wise activation patching."
    )
    parser.add_argument("--model", required=True)  # HuggingFace model id or local path
    parser.add_argument("--model-short", required=True)  # output/filter prefix used by Step 2/3
    parser.add_argument(
        "--langs",
        nargs="+",
        default=DEFAULT_LANGS,
        help="Language codes used to resolve the cross_lang_* filtered directory.",
    )
    parser.add_argument(
        "--lang-pairs",
        nargs="+",
        default=None,
        help="Optional directional language pairs as langA-langB. If omitted, all ordered pairs from --langs are used.",
    )
    parser.add_argument(
        "--patch-type",
        required=True,
        choices=PATCH_TYPES,
        help="Patching type: attn, mlp, or residual.",
    )
    parser.add_argument(
        "--prompt-key",
        default="two_hop",
        choices=PROMPT_KEYS,
        help="Prompt field inside each multilingual_hop record.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=30,
        help="Maximum number of samples per language pair. 0 means no limit.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed",
        help="Step 1 processed data root.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Step 2/3 output root used to resolve filtered pair files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "output" / "step2_5_activation_patching",
        help="Step 2.5 output root.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to load model onto (default: cuda).",
    )
    parser.add_argument(
        "--gpu-index", type=int, default=None,
        help="GPU index to use when device is cuda.",
    )
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 0:
        parser.error("--max-samples must be >= 0")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    logger = setup_logging(f"step2_5_activation_patching_{args.model_short}", LOG_DIR)
    lang_pairs = build_lang_pairs(args.langs, args.lang_pairs)

    logger.info("Loading model=%s model_short=%s", args.model, args.model_short)
    device_str = args.device
    if args.device == "cuda" and args.gpu_index is not None:
        device_str = f"cuda:{args.gpu_index}"
        try:
            torch.cuda.set_device(args.gpu_index)
        except Exception:
            logger.warning(
                "Could not set torch.cuda device to %s; using %s",
                args.gpu_index, device_str,
            )

    model = load_hooked_model(
        args.model,
        device=device_str,
        torch_dtype=args.torch_dtype,
        hf_token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
    )
    n_layers = int(model.cfg.n_layers)
    logger.info("Model loaded. n_layers=%d", n_layers)

    pair_output_dir = (
        args.output_root
        / args.model_short
        / args.patch_type
    )
    pair_output_dir.mkdir(parents=True, exist_ok=True)

    for lang_a, lang_b in lang_pairs:
        logger.info("=" * 60)
        logger.info("Processing: %s → %s (patch_type=%s)", lang_a, lang_b, args.patch_type)
        logger.info("=" * 60)

        try:
            samples, pair_file = load_pairwise_samples(
                data_dir=args.data_dir,
                output_dir=args.input_root,
                model_short=args.model_short,
                langs=args.langs,
                lang_a=lang_a,
                lang_b=lang_b,
                max_samples=args.max_samples,
            )
        except FileNotFoundError as e:
            logger.error(str(e))
            continue

        logger.info("Loaded %d samples from %s", len(samples), pair_file)
        pair_tag = f"{lang_a}_{lang_b}"

        all_results = []
        for sample in tqdm(samples, desc=f"{lang_a}→{lang_b}"):
            result = analyze_sample(
                model, sample, args.prompt_key, args.patch_type, n_layers, logger,
            )
            if result is None:
                continue

            result["patch_type"] = args.patch_type
            result["lang_a"] = lang_a
            result["lang_b"] = lang_b
            result["model"] = args.model_short
            result["source_filtered_file"] = str(pair_file)

            qid = result["id"]
            html_path = pair_output_dir / f"{qid}_{pair_tag}.html"
            title = f"Stage 2 | {args.model_short} | {args.patch_type} | {qid} | {lang_a}→{lang_b}"
            write_html(result, html_path, title)
            all_results.append(result)

        if all_results:
            avg = aggregate_results(all_results)
            avg["patch_type"] = args.patch_type
            avg["lang_a"] = lang_a
            avg["lang_b"] = lang_b
            avg["model"] = args.model_short
            avg["id"] = "average"
            avg["source_filtered_file"] = str(pair_file)
            avg_path = pair_output_dir / f"average_{pair_tag}.html"
            title = f"Stage 2 Average | {args.model_short} | {args.patch_type} | {lang_a}→{lang_b}"
            write_html(avg, avg_path, title)
            logger.info("Saved average HTML: %s", avg_path)

        logger.info("Completed %s → %s: %d samples processed", lang_a, lang_b, len(all_results))

    logger.info("All done.")


if __name__ == "__main__":
    main()
