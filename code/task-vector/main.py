import os
import argparse
import logging
import json
from typing import List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from icl_task_vectors import (
    PromptBuilder,
    TaskVectorConfig,
    TaskVector,
    TaskVectorExtractor,
    Injector,
    choose_backend,
)
from coverage.demontration_pair import DemoPair


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Global config variables
MAX_TOKENS = None
BLOCK_IDX = None
ALPHA = None
AVERAGE_SEPARATORS = None
NORMALIZE = None
SYSTEM_PROMPT = None
TEMPERATURE = None
TOP_K = None
TOP_P = None
NUM_SHOTS = None


def build_fewshot_prompt(pairs: Sequence[DemoPair], query_x: str, sep: str, system_prompt: str | None = None) -> str:
    """Build a compact few-shot prompt where each demo is a single line:
    <input><sep><output>
    and the query is <query_x><sep>
    """
    chunks = [f"{p.input}{sep}{p.output}" for p in pairs]
    chunks.append(f"{query_x}{sep}")
    user_prompt = ",".join(chunks)
    return user_prompt if system_prompt is None else f"{system_prompt}\n{user_prompt}"


def build_query_prompt(query_x: str, sep: str, system_prompt: str | None = None) -> str:
    user_prompt = f"{query_x}{sep}"
    return user_prompt if system_prompt is None else f"{system_prompt}\n{user_prompt}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=str, required=True, help="Local directory containing pretrained model/tokenizer")
    p.add_argument("--config", type=str, required=True, help="Path to config file")
    p.add_argument("--query", type=str, required=True, help="The query input for the task")
    p.add_argument("--sep", type=str, default="->", help="Separator token for prompts (default: '->')")
    return p.parse_args()


def load_config(config_path: str) -> None:
    with open(config_path, 'r') as f:
        config = json.load(f)

    global MAX_TOKENS, BLOCK_IDX, ALPHA, AVERAGE_SEPARATORS, NORMALIZE, SYSTEM_PROMPT, TEMPERATURE, TOP_K, TOP_P, NUM_SHOTS
    MAX_TOKENS = config['max_tokens']
    BLOCK_IDX = config['block_idx']
    ALPHA = config['alpha']
    AVERAGE_SEPARATORS = config['average_separators']
    NORMALIZE = config['normalize']
    SYSTEM_PROMPT = config['system_prompt']
    TEMPERATURE = config['temperature']
    TOP_K = config['top_k']
    TOP_P = config['top_p']
    NUM_SHOTS = config.get('num_shots', 3)


def prepare_environment(model_source: str, local_files_only: bool = True) -> None:
    if local_files_only:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        if not os.path.isdir(model_source):
            raise RuntimeError(f"Model dir {model_source} does not exist or is not a directory.")


def load_model_and_tokenizer(model_source: str, device: str, local_files_only: bool = True):
    logger.info("Loading tokenizer and model from %s", model_source)
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_source, local_files_only=local_files_only, dtype="float16", low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    # Some tokenizers (e.g. GPT-2) have no pad token; set it for generation convenience
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def get_demo_pairs(demos_path: str | None = None) -> List[DemoPair]:
    # Load demo pairs from a JSON file.
    if demos_path is None:
        demos_path = os.path.join(os.path.dirname(__file__), "demos.json")

    with open(demos_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs: List[DemoPair] = []
    for item in data:
        if isinstance(item, dict):
            # Format: {"input": ..., "output": ..., "id": ..., "meta": {}}
            pairs.append(DemoPair(
                input=item['input'],
                output=item['output'],
                id=item['id'],
                meta=item.get('meta', {})
            ))
        else:
            raise ValueError(f"Unsupported demo entry format: {item}. Expected dict with 'input', 'output', 'id', 'meta'.")
    return pairs

def group_demos(demos: Sequence[DemoPair], group_size: int):
    """Yield groups of group_size. Each group is (group_size-1) shots + 1 query (last item is query)."""
    n = len(demos) // group_size
    for i in range(n):
        start = i * group_size
        shots = demos[start : start + (group_size - 1)]
        query = demos[start + (group_size - 1)].input
        yield shots, query


def extract_task_vectors(
    model,
    cfg: TaskVectorConfig,
    pb: PromptBuilder,
    demos: Sequence[DemoPair],
    sep: str,
    system_prompt: str,
    num_shots: int,
) -> TaskVector:
    extractor = TaskVectorExtractor(model, cfg)
    task_vectors = []
    for shots, query in group_demos(demos, group_size=num_shots + 1):
        fewshot_text = build_fewshot_prompt(shots, query, sep, system_prompt=system_prompt)
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


def main():
    args = parse_args()

    load_config(args.config)

    model_source = args.model_dir
    local_files_only = True

    prepare_environment(model_source, local_files_only=local_files_only)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    logger.info("Using device: %s (cuda_available=%s)", device, torch.cuda.is_available())

    model, tokenizer = load_model_and_tokenizer(model_source, device, local_files_only=local_files_only)

    sep = args.sep
    demos = get_demo_pairs()
    system_prompt = SYSTEM_PROMPT

    pb = PromptBuilder(tokenizer, separator_text=sep)

    cfg = TaskVectorConfig(layer_idx=BLOCK_IDX, average_separators=AVERAGE_SEPARATORS, normalize=NORMALIZE, alpha=ALPHA, device=device)

    avg_task_vector = extract_task_vectors(model, cfg, pb, demos, sep, system_prompt, NUM_SHOTS)

    # Query prompt
    query_text = build_query_prompt(args.query, sep, system_prompt=system_prompt)
    logger.info("\nQuery prompt (no demos):\n%s\n", query_text)
    query_enc = pb.encode(query_text, device=device)
    if len(query_enc.separator_positions) == 0:
        raise RuntimeError("No separator found in query prompt; check sep and template.")
    inject_pos = query_enc.separator_positions[-1]

    # Baseline few-shot generation using the last fewshot_enc we created during extraction; if not present fall back to building one.
    # For simplicity, build a fewshot from the first group.
    first_shots, first_query = list(group_demos(demos, group_size=NUM_SHOTS + 1))[0]
    fewshot_text = build_fewshot_prompt(first_shots, first_query, sep, system_prompt=system_prompt)
    fewshot_enc = pb.encode(fewshot_text, device=device)
    base_few_text = generate_text(model, tokenizer, fewshot_enc)
    base_text = generate_text(model, tokenizer, query_enc)
    logger.info("Baseline (few-shot):\n%s", base_few_text)

    backend = choose_backend(model)
    injector = Injector(backend, cfg)
    # Perform the forward-pass injection hook (keeps behavior similar to prior script)
    injector.inject_and_forward(
        model=model, prompt=query_enc, task_vector=avg_task_vector, inject_position=inject_pos, use_cache=False
    )

    inj_text = install_hooks_and_generate(model, backend, avg_task_vector, inject_pos, query_enc, tokenizer)

    logger.info("\nBaseline (zeroshot without injection):\n%s\n", base_text)
    logger.info("\nInjected:\n%s\n", inj_text)


if __name__ == "__main__":
    main()

