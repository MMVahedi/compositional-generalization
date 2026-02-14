import os
import argparse
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from icl_task_vectors import PromptBuilder

from task_vector_pkg.config import Config, load_config
from task_vector_pkg.demos import get_demo_pairs, group_demos
from task_vector_pkg.dataset import Dataset
from task_vector_pkg.experiment import Experiment

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=str, required=True, help="Local directory containing pretrained model/tokenizer")
    p.add_argument("--config", type=str, required=True, help="Path to config file")
    p.add_argument("--sep", type=str, default="->", help="Separator token for prompts (default: '->')")
    return p.parse_args()


def load_configs(args: argparse.Namespace) -> Config:
    config_dict = load_config(args.config)
    return Config(
        max_tokens=config_dict['max_tokens'],
        block_idx=config_dict['block_idx'],
        alpha=config_dict['alpha'],
        average_separators=config_dict['average_separators'],
        normalize=config_dict['normalize'],
        system_prompt=config_dict['system_prompt'],
        temperature=config_dict['temperature'],
        top_k=config_dict['top_k'],
        top_p=config_dict['top_p'],
        num_shots=config_dict.get('num_shots', 3),
        sep=args.sep
    )


def load_dataset(configs: Config, prompt_builder) -> Dataset:
    demos = get_demo_pairs()
    queries = list(group_demos(demos, group_size=configs.num_shots + 1, tokenizer=prompt_builder.tokenizer))
    return Dataset(queries, 0)  # coverage_degree placeholder


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

    # Load configs
    configs = load_configs(args)

    model_source = args.model_dir
    local_files_only = True

    prepare_environment(model_source, local_files_only=local_files_only)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    logger.info("Using device: %s (cuda_available=%s)", device, torch.cuda.is_available())

    model, tokenizer = load_model_and_tokenizer(model_source, device, local_files_only=local_files_only)

    prompt_builder = PromptBuilder(tokenizer, separator_text=configs.sep)

    # Load dataset
    dataset = load_dataset(configs, prompt_builder)

    # Create and run experiment
    experiment = Experiment(dataset, model, configs, prompt_builder)
    results = experiment.run()

    logger.info("Experiment results: %s", results)


if __name__ == "__main__":
    main()

