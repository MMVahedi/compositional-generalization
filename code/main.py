import os
import argparse

from task_vector.utils import Config
from dataset.dataset import DatasetBuilder

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=str, required=True, help="Local directory containing pretrained model/tokenizer")
    p.add_argument("--config", type=str, required=True, help="Path to config file")
    p.add_argument("--degree", type=int, required=True, help="Degree of coverage for dataset construction")
    return p.parse_args()


def prepare_environment(model_source: str, local_files_only: bool = True) -> None:
    if local_files_only:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        if not os.path.isdir(model_source):
            raise RuntimeError(f"Model dir {model_source} does not exist or is not a directory.")


def main():
    args = parse_args()

    # Load configs
    configs = Config.load_config(args.config)

    prepare_environment(args.model_dir, local_files_only=True)

    # Load demos
    get_demo_pairs(args.demos_path)
    demos = DemoPair.all_instances
    builder = DatasetBuilder(demos, num_shots=configs.num_shots)
    dataset = builder.get_dataset(coverage_degree=1)

    # Create and run experiment
    experiment = TaskVectorExperiment(dataset, model, configs)
    results = experiment.run()

    logger.info("Experiment results: %s", results)


if __name__ == "__main__":
    main()

