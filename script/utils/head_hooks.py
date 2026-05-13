"""head_hooks.py — Model-side hooks and NLL helpers for Bridge Head analysis.

Provides:
  - HeadMaskManager: registers per-head scalar mask hooks on o_proj for
    gradient-based head importance scoring, mean ablation, and scaling.
  - compute_nll_with_grad: teacher-forced NLL + gradient extraction.
  - compute_nll_eval: teacher-forced NLL without gradients (for ablation eval).
  - collect_mean_head_outputs: collects mean per-head output vectors for
    mean-ablation calibration.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from utils.prompt_utils import wrap_prompt

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_o_proj(model: nn.Module, layer_idx: int) -> nn.Module | None:
    """Return the o_proj Linear of the given transformer layer, or None."""
    # Llama / Qwen style: model.model.layers[i].self_attn.o_proj
    try:
        layers = model.model.layers
        return layers[layer_idx].self_attn.o_proj
    except (AttributeError, IndexError):
        pass
    # Fallback: model.layers[i].self_attn.o_proj
    try:
        layers = model.layers
        return layers[layer_idx].self_attn.o_proj
    except (AttributeError, IndexError):
        return None


def _get_num_layers(model: nn.Module) -> int:
    try:
        return model.config.num_hidden_layers
    except AttributeError:
        pass
    try:
        return len(model.model.layers)
    except AttributeError:
        return len(model.layers)


def _get_num_heads(model: nn.Module) -> int:
    """Return the number of query attention heads (n_q_heads for GQA models)."""
    return model.config.num_attention_heads


def _get_head_dim(model: nn.Module) -> int:
    hidden = model.config.hidden_size
    n_heads = _get_num_heads(model)
    return hidden // n_heads


# ---------------------------------------------------------------------------
# HeadMaskManager
# ---------------------------------------------------------------------------

class HeadMaskManager:
    """Manages per-head scalar mask parameters attached to o_proj inputs.

    The mask is applied as a pre-hook on the o_proj linear layer:
        input[:, h * head_dim : (h+1) * head_dim] *= mask[h]
    This is mathematically equivalent to scaling the post-W_O contribution
    of each head in the residual stream.

    Supports gradient-based scoring, mean ablation, and alpha scaling.

    Parameters
    ----------
    model:
        HuggingFace CausalLM model.  All model parameters are frozen inside
        this class; only the mask parameters receive gradients.
    n_layers:
        Number of transformer layers.
    n_heads:
        Number of query attention heads per layer.
    head_dim:
        Dimensionality of each head.  Computed from model config if not given.
    """

    def __init__(
        self,
        model: nn.Module,
        n_layers: int | None = None,
        n_heads: int | None = None,
        head_dim: int | None = None,
    ) -> None:
        self.model = model
        self.n_layers = n_layers or _get_num_layers(model)
        self.n_heads = n_heads or _get_num_heads(model)
        self.head_dim = head_dim or _get_head_dim(model)

        # Freeze all model parameters.
        for p in model.parameters():
            p.requires_grad_(False)

        # One mask tensor per layer; device determined at hook-registration time.
        # Store as plain tensors first; converted to Parameters when registered.
        self._masks: list[nn.Parameter] = []
        self._hooks: list[Any] = []

    # ------------------------------------------------------------------
    # Hook registration / cleanup
    # ------------------------------------------------------------------

    def register_hooks(self) -> None:
        """Attach forward pre-hooks to each layer's o_proj."""
        if self._hooks:
            self.remove_hooks()

        self._masks = []
        for layer_idx in range(self.n_layers):
            o_proj = _get_o_proj(self.model, layer_idx)
            if o_proj is None:
                raise RuntimeError(
                    f"Could not locate o_proj for layer {layer_idx}. "
                    "Ensure the model uses Llama/Qwen-style naming."
                )
            device = o_proj.weight.device
            mask = nn.Parameter(
                torch.ones(self.n_heads, dtype=torch.float32, device=device),
                requires_grad=True,
            )
            self._masks.append(mask)

            # Closure captures layer_idx and mask.
            def _make_hook(m: nn.Parameter, n_h: int, h_dim: int):
                def hook(module: nn.Module, inputs: tuple) -> tuple:
                    x = inputs[0]  # shape: [B*S, n_heads * head_dim] or [B, S, ...]
                    orig_shape = x.shape
                    # Flatten to [..., n_heads * head_dim]
                    flat = x.reshape(-1, n_h * h_dim)
                    # Apply per-head scalar mask
                    scale = m.to(flat.dtype)  # [n_heads]
                    # Expand to [1, n_heads * head_dim] via repeat_interleave
                    scale_expanded = scale.repeat_interleave(h_dim).unsqueeze(0)  # [1, n_h*h_dim]
                    flat = flat * scale_expanded
                    return (flat.reshape(orig_shape),) + inputs[1:]
                return hook

            h = o_proj.register_forward_pre_hook(
                _make_hook(mask, self.n_heads, self.head_dim)
            )
            self._hooks.append(h)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Mask manipulation
    # ------------------------------------------------------------------

    def reset_masks(self) -> None:
        """Re-initialise all mask values to 1.0 (identity)."""
        for mask in self._masks:
            with torch.no_grad():
                mask.fill_(1.0)

    def zero_grads(self) -> None:
        """Zero accumulated gradients on all mask parameters."""
        for mask in self._masks:
            if mask.grad is not None:
                mask.grad.zero_()

    def set_ablation(self, layer: int, head: int, value: float) -> None:
        """Set the mask for a single head to *value*.

        ``value=0.0`` fully ablates the head; ``value=mean_contribution`` for
        mean ablation.
        """
        with torch.no_grad():
            self._masks[layer][head] = value

    def set_scaling(self, layer: int, head: int, alpha: float) -> None:
        """Scale a head by ``(1 + alpha)``.  ``alpha > 0`` amplifies."""
        with torch.no_grad():
            self._masks[layer][head] = 1.0 + alpha

    def apply_head_set_ablation(
        self,
        head_set: list[tuple[int, int]],
        mean_values: dict[tuple[int, int], float],
    ) -> None:
        """Batch-ablate a list of ``(layer, head)`` pairs using their mean values."""
        for layer, head in head_set:
            self.set_ablation(layer, head, mean_values.get((layer, head), 0.0))

    def apply_head_set_scaling(
        self,
        head_set: list[tuple[int, int]],
        alpha: float,
    ) -> None:
        """Batch-scale a list of ``(layer, head)`` pairs by ``(1 + alpha)``."""
        for layer, head in head_set:
            self.set_scaling(layer, head, alpha)

    # ------------------------------------------------------------------
    # Gradient extraction
    # ------------------------------------------------------------------

    def get_gradients(self) -> np.ndarray:
        """Return ``abs(grad)`` for every head as ``np.ndarray[n_layers, n_heads]``."""
        grads = np.zeros((self.n_layers, self.n_heads), dtype=np.float32)
        for layer_idx, mask in enumerate(self._masks):
            if mask.grad is not None:
                grads[layer_idx] = mask.grad.abs().detach().float().cpu().numpy()
        return grads

    # ------------------------------------------------------------------
    # Context manager convenience
    # ------------------------------------------------------------------

    def __enter__(self) -> "HeadMaskManager":
        self.register_hooks()
        return self

    def __exit__(self, *_) -> None:
        self.remove_hooks()


# ---------------------------------------------------------------------------
# NLL helpers
# ---------------------------------------------------------------------------

def _build_inputs_and_labels(
    tokenizer: Any,
    prompt_text: str,
    gold_text: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Tokenise prompt + gold jointly and return (input_ids, labels).

    Labels are ``-100`` for prompt tokens and the actual token IDs for gold
    tokens.  Returns ``None`` if either prompt or gold tokenises to empty.
    """
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    gold_ids = tokenizer.encode(gold_text, add_special_tokens=False)
    if not prompt_ids or not gold_ids:
        return None

    full_ids = prompt_ids + gold_ids
    labels = [-100] * len(prompt_ids) + gold_ids

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    label_ids = torch.tensor([labels], dtype=torch.long, device=device)
    return input_ids, label_ids


def _first_param_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def compute_nll_with_grad(
    model: nn.Module,
    tokenizer: Any,
    prompt_text: str,
    gold_text: str,
    mask_manager: HeadMaskManager,
) -> tuple[float, np.ndarray]:
    """Compute teacher-forced NLL and extract per-head gradient scores.

    Parameters
    ----------
    prompt_text:
        Already-wrapped prompt (e.g. output of ``wrap_prompt()``).
    gold_text:
        Clean gold answer span.
    mask_manager:
        A :class:`HeadMaskManager` with hooks already registered.

    Returns
    -------
    (nll, scores)
        *nll* is the mean NLL over gold tokens.
        *scores* is ``np.ndarray[n_layers, n_heads]`` of ``abs(grad)`` values.
    """
    device = _first_param_device(model)
    result = _build_inputs_and_labels(tokenizer, prompt_text, gold_text, device)
    if result is None:
        return float("nan"), np.zeros((mask_manager.n_layers, mask_manager.n_heads), dtype=np.float32)

    input_ids, label_ids = result
    mask_manager.zero_grads()

    outputs = model(input_ids=input_ids, labels=label_ids)
    nll = outputs.loss
    nll.backward()

    scores = mask_manager.get_gradients()
    mask_manager.zero_grads()
    return float(nll.detach().cpu()), scores


@torch.no_grad()
def compute_nll_eval(
    model: nn.Module,
    tokenizer: Any,
    prompt_text: str,
    gold_text: str,
) -> float:
    """Teacher-forced NLL without gradients (for ablation/amplification eval)."""
    device = _first_param_device(model)
    result = _build_inputs_and_labels(tokenizer, prompt_text, gold_text, device)
    if result is None:
        return float("nan")
    input_ids, label_ids = result
    outputs = model(input_ids=input_ids, labels=label_ids)
    return float(outputs.loss.cpu())


# ---------------------------------------------------------------------------
# Mean head output collection (calibration for mean ablation)
# ---------------------------------------------------------------------------

def collect_mean_head_outputs(
    model: nn.Module,
    tokenizer: Any,
    prompts: list[str],
    n_layers: int,
    n_heads: int,
    head_dim: int,
) -> dict[tuple[int, int], float]:
    """Collect the mean scalar activation entering o_proj for each head.

    We record the mean *norm* of the head's pre-o_proj slice across all tokens
    and examples.  This scalar is used as the ablation value so that
    ``mask * head_output`` mimics the expected (mean) contribution.

    Returns
    -------
    dict mapping ``(layer, head)`` to mean scalar value.
    """
    # Accumulators: sum and count per (layer, head)
    sums = np.zeros((n_layers, n_heads), dtype=np.float64)
    counts = np.zeros((n_layers, n_heads), dtype=np.int64)
    hooks = []

    def _make_capture_hook(layer_idx: int, n_h: int, h_dim: int):
        def hook(module: nn.Module, inputs: tuple) -> None:
            x = inputs[0].detach().float()  # [B*S, n_h*h_dim] or [B, S, n_h*h_dim]
            flat = x.reshape(-1, n_h * h_dim).cpu().numpy()  # [T, n_h*h_dim]
            for h in range(n_h):
                head_slice = flat[:, h * h_dim : (h + 1) * h_dim]
                sums[layer_idx, h] += float(np.abs(head_slice).mean())
                counts[layer_idx, h] += 1
        return hook

    # Attach capture hooks
    for layer_idx in range(n_layers):
        o_proj = _get_o_proj(model, layer_idx)
        if o_proj is None:
            continue
        hooks.append(
            o_proj.register_forward_pre_hook(
                _make_capture_hook(layer_idx, n_heads, head_dim)
            )
        )

    device = _first_param_device(model)
    with torch.no_grad():
        for prompt_text in prompts:
            ids = tokenizer.encode(prompt_text, add_special_tokens=True, return_tensors="pt")
            model(input_ids=ids.to(device))

    for h in hooks:
        h.remove()

    mean_vals: dict[tuple[int, int], float] = {}
    for l in range(n_layers):
        for h in range(n_heads):
            c = counts[l, h]
            mean_vals[(l, h)] = float(sums[l, h] / c) if c > 0 else 0.0
    return mean_vals
