from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, Tuple, Union

import torch
import torch.nn as nn


# =========================
# Config + Data containers
# =========================

@dataclass
class TaskVectorConfig:
    # extraction
    layer_idx: int = -1                   # which hidden_states index to read (supports negative)
    average_separators: bool = False      # False => last separator only; True => mean over all
    normalize: Optional[str] = None       # None | "l2"

    # injection
    alpha: float = 1.0
    device: Union[str, torch.device] = "cpu"


@dataclass
class TaskVector:
    vector: torch.Tensor                  # [d_model]
    layer_idx: int
    separator_text: str
    average_separators: bool
    meta: dict


@dataclass
class EncodedPrompt:
    input_ids: torch.Tensor               # [1, T]
    attention_mask: Optional[torch.Tensor]
    separator_positions: List[int]        # token indices where we consider the "separator embedding"


# =========================
# Prompt building (find separator token positions)
# =========================

def find_subsequence_starts(sequence: List[int], subseq: List[int]) -> List[int]:
    """Return start indices where subseq appears in sequence."""
    if len(subseq) == 0 or len(sequence) < len(subseq):
        return []
    hits = []
    m = len(subseq)
    for i in range(0, len(sequence) - m + 1):
        if sequence[i:i + m] == subseq:
            hits.append(i)
    return hits


class PromptBuilder:
    """
    Locates separator occurrences robustly even when separator_text is multi-token.
    We define the separator position as the *last token* of the separator subsequence.
    """
    def __init__(self, tokenizer, separator_text: str):
        self.tokenizer = tokenizer
        self.separator_text = separator_text
        self.sep_token_ids = tokenizer.encode(separator_text, add_special_tokens=False)
        if len(self.sep_token_ids) == 0:
            raise ValueError("separator_text produced 0 tokens; choose a different separator.")

    def encode(self, text: str, device: Union[str, torch.device] = "cpu") -> EncodedPrompt:
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        seq = input_ids[0].tolist()
        starts = find_subsequence_starts(seq, self.sep_token_ids)
        # Use last token index of the separator subsequence
        positions = [s + (len(self.sep_token_ids) - 1) for s in starts]
        return EncodedPrompt(input_ids=input_ids, attention_mask=attention_mask, separator_positions=positions)


# =========================
# Extract task vector from hidden states
# =========================

class TaskVectorExtractor:
    def __init__(self, model: nn.Module, cfg: TaskVectorConfig):
        self.model = model
        self.cfg = cfg

    @torch.no_grad()
    def extract(self, prompt: EncodedPrompt, separator_text: str) -> TaskVector:
        if not prompt.separator_positions:
            raise ValueError("No separator occurrences found. Check your separator_text and prompt template.")

        outputs = self.model(
            input_ids=prompt.input_ids,
            attention_mask=prompt.attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        hidden_states = outputs.hidden_states
        hs = hidden_states[self.cfg.layer_idx]  # [1, T, d_model]

        if self.cfg.average_separators:
            pos = prompt.separator_positions
        else:
            pos = [prompt.separator_positions[-1]]

        vecs = hs[0, pos, :]        # [k, d_model]
        v = vecs.mean(dim=0)        # [d_model]

        if self.cfg.normalize == "l2":
            v = v / (v.norm(p=2) + 1e-12)

        return TaskVector(
            vector=v.detach(),
            layer_idx=self.cfg.layer_idx,
            separator_text=separator_text,
            average_separators=self.cfg.average_separators,
            meta={"positions_used": pos, "seq_len": int(prompt.input_ids.shape[-1])},
        )


# =========================
# Injection backends (one per model family)
# =========================

class HookHandle(Protocol):
    def remove(self) -> None: ...


class InjectorBackend(Protocol):
    def install_hook(
        self,
        model: nn.Module,
        layer_idx: int,
        position: int,
        add_vector: torch.Tensor,
        alpha: float,
    ) -> HookHandle:
        ...


def _resolve_layer_index(n_layers: int, layer_idx: int) -> int:
    """Convert possibly-negative index into [0, n_layers-1]."""
    if layer_idx < 0:
        layer_idx = n_layers + layer_idx
    if not (0 <= layer_idx < n_layers):
        raise IndexError(f"layer_idx out of range: got {layer_idx} for n_layers={n_layers}")
    return layer_idx


def _add_at_position(x: torch.Tensor, position: int, add_vec: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    x: [B, T, D]
    add_vec: [D] (or [1,1,D])
    Adds alpha*add_vec at token index `position` for all batch items.
    """
    if x.dim() != 3:
        return x
    B, T, D = x.shape
    if position < 0:
        position = T + position
    if not (0 <= position < T):
        # out-of-range => do nothing (you can set strict if you prefer)
        return x

    v = add_vec
    if v.dim() == 1:
        v = v.view(1, 1, -1)
    v = v.to(device=x.device, dtype=x.dtype)

    x = x.clone()  # avoid in-place on activations used elsewhere
    x[:, position:position + 1, :] = x[:, position:position + 1, :] + alpha * v
    return x


# ---- GPT-2 style: model.transformer.h is a ModuleList of blocks ----
class GPT2StyleBackend:
    def _blocks(self, model: nn.Module) -> nn.ModuleList:
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h
        raise TypeError("GPT2StyleBackend could not find model.transformer.h")

    def install_hook(self, model, layer_idx, position, add_vector, alpha):
        blocks = self._blocks(model)
        idx = _resolve_layer_index(len(blocks), layer_idx)

        def hook_fn(module, inputs, output):
            # GPT-2 blocks typically return tensor [B,T,D]
            if torch.is_tensor(output):
                return _add_at_position(output, position, add_vector, alpha)
            # Some blocks return tuples; try first element as hidden states.
            if isinstance(output, tuple) and torch.is_tensor(output[0]):
                new0 = _add_at_position(output[0], position, add_vector, alpha)
                return (new0,) + output[1:]
            return output

        return blocks[idx].register_forward_hook(hook_fn)


# ---- LLaMA / Qwen2-style (HF): model.model.layers is ModuleList ----
class LlamaBackend:
    def _blocks(self, model: nn.Module) -> nn.ModuleList:
        # HF LLaMA-like: model.model.layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        raise TypeError("LlamaBackend could not find model.model.layers")

    def install_hook(self, model, layer_idx, position, add_vector, alpha):
        blocks = self._blocks(model)
        idx = _resolve_layer_index(len(blocks), layer_idx)

        def hook_fn(module, inputs, output):
            # LLaMA blocks often return tuple(hidden_states, ...)
            if torch.is_tensor(output):
                return _add_at_position(output, position, add_vector, alpha)
            if isinstance(output, tuple) and torch.is_tensor(output[0]):
                new0 = _add_at_position(output[0], position, add_vector, alpha)
                return (new0,) + output[1:]
            return output

        return blocks[idx].register_forward_hook(hook_fn)


class QwenBackend:
    """
    Qwen models vary across generations:
      - Qwen2 (HF) is typically LLaMA-like: model.model.layers
      - Some older Qwen variants may expose transformer.h
    This backend tries LLaMA-style first, then GPT2-style.
    """
    def __init__(self):
        self._llama = LlamaBackend()
        self._gpt2 = GPT2StyleBackend()

    def install_hook(self, model, layer_idx, position, add_vector, alpha):
        try:
            return self._llama.install_hook(model, layer_idx, position, add_vector, alpha)
        except TypeError:
            return self._gpt2.install_hook(model, layer_idx, position, add_vector, alpha)


# =========================
# Injector
# =========================

class Injector:
    def __init__(self, backend: InjectorBackend, cfg: TaskVectorConfig):
        self.backend = backend
        self.cfg = cfg

    def inject_and_forward(
        self,
        model: nn.Module,
        prompt: EncodedPrompt,
        task_vector: TaskVector,
        inject_position: int,
        **forward_kwargs,
    ):
        add_vec = task_vector.vector.to(self.cfg.device)
        handle = self.backend.install_hook(
            model=model,
            layer_idx=task_vector.layer_idx,
            position=inject_position,
            add_vector=add_vec,
            alpha=self.cfg.alpha,
        )
        try:
            return model(
                input_ids=prompt.input_ids,
                attention_mask=prompt.attention_mask,
                **forward_kwargs,
            )
        finally:
            handle.remove()


# =========================
# Backend selection helper
# =========================

def choose_backend(model: nn.Module, prefer: Optional[str] = None) -> InjectorBackend:
    """
    prefer: "llama" | "qwen" | "gpt2" | None
    If None, infer by attribute structure.
    """
    if prefer == "gpt2":
        return GPT2StyleBackend()
    if prefer == "llama":
        return LlamaBackend()
    if prefer == "qwen":
        return QwenBackend()

    # auto-infer
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        # LLaMA-like (also Qwen2)
        return LlamaBackend()
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return GPT2StyleBackend()

    # fallback: QwenBackend tries both
    return QwenBackend()

