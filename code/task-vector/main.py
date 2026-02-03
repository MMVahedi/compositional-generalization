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

from task_vector_pkg.config import load_config, BLOCK_IDX, ALPHA, AVERAGE_SEPARATORS, NORMALIZE, SYSTEM_PROMPT, TEMPERATURE, TOP_K, TOP_P, NUM_SHOTS
from task_vector_pkg.prompts import build_query_prompt
from task_vector_pkg.demos import get_demo_pairs, group_demos
from task_vector_pkg.task_vector import TaskVectorBuilder
from task_vector_pkg.dataset import Dataset
from task_vector_pkg.experiment import Experiment

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

    configs = {
        'max_tokens': MAX_TOKENS,
        'block_idx': BLOCK_IDX,
        'alpha': ALPHA,
        'average_separators': AVERAGE_SEPARATORS,
        'normalize': NORMALIZE,
        'system_prompt': SYSTEM_PROMPT,
        'temperature': TEMPERATURE,
        'top_k': TOP_K,
        'top_p': TOP_P,
        'num_shots': NUM_SHOTS,
        'sep': args.sep
    }

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
    dataset = Dataset(queries, 0)  # coverage_degree placeholder
    experiment = Experiment(dataset, model, configs, prompt_builder)
    
    # Run the experiment
    results = experiment.run()
    logger.info("Experiment results: %s", results)


if __name__ == "__main__":
    main()

