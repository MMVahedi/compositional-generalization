from __future__ import annotations

import logging
from typing import Protocol

import torch
import torch.nn as nn
from task_vector.task_vector_prompt import TaskVectorPrompt
from task_vector.utils import TaskVectorConfig, TaskVector

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
    x[:, position:position + 1, :] = (1-alpha)*x[:, position:position + 1, :] + alpha * v
    return x


# ---- GPT-2 style: model.transformer.h is a ModuleList of blocks ----
class GPT2StyleBackend:
    def _blocks(self, model: nn.Module) -> nn.ModuleList:
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h
        raise TypeError("GPT2StyleBackend could not find model.transformer.h")

    def install_hook(self, model, layer_idx, position, add_vector, alpha):
        blocks = self._blocks(model)
        idx = layer_idx

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
        idx = layer_idx

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
    def __init__(self, backend: InjectorBackend, task_vector: TaskVector, cfg: TaskVectorConfig):
        logging.debug(f"Initializing Injector with backend={backend.__class__.__name__}")
        self.backend = backend
        self.task_vector = task_vector
        self.cfg = cfg
        logging.debug(f"Injector config: alpha={cfg.alpha}, layer_idx={cfg.layer_idx}")

    def inject_and_forward(
        self,
        model: nn.Module,
        prompt: TaskVectorPrompt,
        inject_position: int,
        **forward_kwargs,
    ):
        logging.debug(f"Starting injection at position={inject_position}")
        # Support multiple layer indices and per-layer vectors.
        # Normalize layer indices and vectors to lists of same length.
        if isinstance(self.task_vector.layer_idx, int):
            layer_idxs = [self.task_vector.layer_idx]
        else:
            layer_idxs = list(self.task_vector.layer_idx)
        
        logging.debug(f"Injecting into {len(layer_idxs)} layer(s): {layer_idxs}")

        tv = self.task_vector.vector
        per_layer_vecs = []
        if tv.dim() == 1:
            # same vector for all layers
            per_layer_vecs = [tv] * len(layer_idxs)
        elif tv.dim() == 2:
            if tv.shape[0] != len(layer_idxs):
                raise ValueError("Number of per-layer vectors does not match number of layer indices")
            per_layer_vecs = [tv[i] for i in range(tv.shape[0])]
        else:
            raise ValueError("Unsupported task_vector.vector shape for multi-layer injection")

        handles = []
        try:
            logging.debug(f"Installing {len(layer_idxs)} hook(s) for injection")
            for li, v in zip(layer_idxs, per_layer_vecs):
                add_vec = v.to(self.cfg.device)
                h = self.backend.install_hook(
                    model=model,
                    layer_idx=li,
                    position=inject_position,
                    add_vector=add_vec,
                    alpha=self.cfg.alpha,
                )
                handles.append(h)
            
            logging.debug("Running forward pass with injected vectors")
            result = model.generate(
                input_ids=prompt.input_ids,
                attention_mask=prompt.attention_mask,
                **forward_kwargs,
            )
            logging.debug("Forward pass completed")
            return result
        finally:
            logging.debug(f"Removing {len(handles)} hook(s)")
            for h in handles:
                h.remove()
