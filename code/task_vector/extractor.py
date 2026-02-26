from __future__ import annotations


import torch
import torch.nn as nn
from task_vector.task_vector_prompt import TaskVectorPrompt
from task_vector.utils import TaskVectorConfig, TaskVector


class TaskVectorExtractor:
    def __init__(self, model: nn.Module, cfg: TaskVectorConfig):
        self.model = model
        self.cfg = cfg

    @torch.no_grad()
    def extract(self, prompt: TaskVectorPrompt, separator_text: str) -> TaskVector:
        if not prompt.separator_positions:
            raise ValueError("No separator occurrences found. Check your separator_text and prompt template.")

        outputs = self.model(
            input_ids=prompt.input_ids,
            attention_mask=prompt.attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        hidden_states = outputs.hidden_states

        # Normalize layer indices into a list for uniform handling
        cfg_layer_idx = self.cfg.layer_idx
        if isinstance(cfg_layer_idx, int):
            layer_idxs = [cfg_layer_idx]
        else:
            layer_idxs = list(cfg_layer_idx)

        # choose separator token positions once (same for all layers)
        if self.cfg.average_separators:
            positions = prompt.separator_positions
        else:
            positions = [prompt.separator_positions[-1]]

        per_layer_vecs = []
        resolved_layers = []
        for li in layer_idxs:
            ridx = li
            resolved_layers.append(ridx)
            hs_layer = hidden_states[ridx]  # [1, T, d_model]
            vecs = hs_layer[0, positions, :]  # [k, d_model]
            v = vecs.mean(dim=0)             # [d_model]
            if self.cfg.normalize == "l2":
                v = v / (v.norm(p=2) + 1e-12)
            per_layer_vecs.append(v)

        # Stack per-layer vectors. If only one layer, return 1D tensor for compatibility.
        if len(per_layer_vecs) == 1:
            final_vec = per_layer_vecs[0].detach()
        else:
            final_vec = torch.stack(per_layer_vecs, dim=0).detach()  # [n_layers, d_model]

        return TaskVector(
            vector=final_vec,
            layer_idx=layer_idxs if len(layer_idxs) > 1 else layer_idxs[0],
            separator_text=separator_text,
            average_separators=self.cfg.average_separators,
            meta={
                "positions_used": positions,
                "seq_len": int(prompt.input_ids.shape[-1]),
                "layers_used": resolved_layers,
            },
        )
