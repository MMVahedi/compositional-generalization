import os
import argparse
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from icl_task_vectors import (
    PromptBuilder,
    TaskVectorConfig,
    Injector,
    choose_backend,
)

from task_vector_pkg.config import load_config, BLOCK_IDX, ALPHA, AVERAGE_SEPARATORS, NORMALIZE, SYSTEM_PROMPT, NUM_SHOTS
from task_vector_pkg.prompts import build_query_prompt
from task_vector_pkg.demos import get_demo_pairs, group_demos
from task_vector_pkg.task_vector import TaskVectorBuilder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=str, required=True, help="Local directory containing pretrained model/tokenizer")
    p.add_argument("--config", type=str, required=True, help="Path to config file")
    p.add_argument("--query", type=str, required=True, help="The query input for the task")
    p.add_argument("--sep", type=str, default="->", help="Separator token for prompts (default: '->')")
    return p.parse_args()


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

    prompt_builder = PromptBuilder(tokenizer, separator_text=sep)

    cfg = TaskVectorConfig(layer_idx=BLOCK_IDX, average_separators=AVERAGE_SEPARATORS, normalize=NORMALIZE, alpha=ALPHA, device=device)

    queries = list(group_demos(demos, group_size=NUM_SHOTS + 1, tokenizer=prompt_builder.tokenizer))
    tv_builder = TaskVectorBuilder(model, cfg, prompt_builder)
    avg_task_vector = tv_builder.build_task_vector(queries, sep, system_prompt)

    # Query prompt
    query_text = build_query_prompt(args.query, sep, system_prompt=system_prompt)
    logger.info("\nQuery prompt (no demos):\n%s\n", query_text)
    query_enc = prompt_builder.encode(query_text, device=device)
    if len(query_enc.separator_positions) == 0:
        raise RuntimeError("No separator found in query prompt; check sep and template.")
    inject_pos = query_enc.separator_positions[-1]

    # Baseline few-shot generation using the last fewshot_enc we created during extraction; if not present fall back to building one.
    # For simplicity, build a fewshot from the first group.
    first_query_obj = list(group_demos(demos, group_size=NUM_SHOTS + 1, tokenizer=prompt_builder.tokenizer))[0]
    fewshot_text = first_query_obj.build_prompt(sep, system_prompt)
    fewshot_enc = prompt_builder.encode(fewshot_text, device=device)
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

