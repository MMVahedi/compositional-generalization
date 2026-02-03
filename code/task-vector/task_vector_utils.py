from coverage.demontration_pair import DemoPair
import torch
from typing import Sequence

from icl_task_vectors import (
    PromptBuilder,
    TaskVectorConfig,
    TaskVector,
    TaskVectorExtractor,
)

from config_loader import MAX_TOKENS, TEMPERATURE, TOP_K, TOP_P


def extract_task_vectors(
    model,
    cfg: TaskVectorConfig,
    pb: PromptBuilder,
    demos: Sequence[DemoPair],
    sep: str,
    system_prompt: str,
    num_shots: int,
) -> TaskVector:
    from prompt_utils import build_fewshot_prompt
    from demo_utils import group_demos
    import logging

    logger = logging.getLogger(__name__)

    extractor = TaskVectorExtractor(model, cfg)
    task_vectors = []
    tokenizer = pb.tokenizer  # Assuming PromptBuilder has tokenizer attribute
    for query_obj in group_demos(demos, group_size=num_shots + 1, tokenizer=tokenizer):
        fewshot_text = query_obj.build_prompt(sep, system_prompt)
        logger.info("Few-shot prompt used for extraction:\n%s\n", fewshot_text)
        fewshot_enc = pb.encode(fewshot_text, device=cfg.device)
        task_vec = extractor.extract(fewshot_enc, separator_text=sep)
        task_vectors.append(task_vec)

    if not task_vectors:
        raise RuntimeError("No task vectors extracted; check your demo grouping and separators.")

    vecs = torch.stack([tv.vector for tv in task_vectors], dim=0)
    avg_vec = vecs.mean(dim=0)
    if cfg.normalize == "l2":
        avg_vec = avg_vec / (avg_vec.norm(p=2) + 1e-12)

    avg_task_vector = TaskVector(
        vector=avg_vec.detach(),
        layer_idx=task_vectors[0].layer_idx,
        separator_text=sep,
        average_separators=cfg.average_separators,
        meta={"n_vectors": len(task_vectors)},
    )
    logger.info("Averaged task vector meta: %s", avg_task_vector.meta)
    return avg_task_vector


def generate_text(model, tokenizer, enc, max_new_tokens: int = None):
    if max_new_tokens is None:
        max_new_tokens = MAX_TOKENS
    out = model.generate(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def install_hooks_and_generate(model, backend, avg_task_vector: TaskVector, inject_pos: int, query_enc, tokenizer):
    from config_loader import ALPHA
    # Prepare per-layer vectors
    tv = avg_task_vector.vector
    layer_idxs = [avg_task_vector.layer_idx] if isinstance(avg_task_vector.layer_idx, int) else list(avg_task_vector.layer_idx)
    if tv.dim() == 1:
        per_layer_vecs = [tv] * len(layer_idxs)
    elif tv.dim() == 2:
        if tv.shape[0] != len(layer_idxs):
            raise RuntimeError("Averaged task vector has mismatched per-layer shape")
        per_layer_vecs = [tv[i] for i in range(tv.shape[0])]
    else:
        raise RuntimeError("Unsupported averaged task vector shape for injection")

    handles = []
    try:
        for li, v in zip(layer_idxs, per_layer_vecs):
            h = backend.install_hook(model=model, layer_idx=li, position=inject_pos, add_vector=v, alpha=ALPHA)
            handles.append(h)
        return generate_text(model, tokenizer, query_enc)
    finally:
        for h in handles:
            h.remove()