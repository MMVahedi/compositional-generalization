import os
import argparse
import logging

from task_vector.utils import Config
from dataset.dataset import DatasetBuilder
from function.twohop import generate_two_hop_function
from experiment.experiment import TaskVectorExperiment

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=str, required=True, help="Local directory containing pretrained model/tokenizer")
    p.add_argument("--config", type=str, required=True, help="Path to config file")
    p.add_argument("--degree", type=int, required=True, help="Degree of coverage for dataset construction")
    return p.parse_args()


def prepare_environment(model_source: str, local_files_only: bool = True) -> None:
    logging.info("Preparing environment...")
    if local_files_only:
        logging.info("Setting offline mode for transformers")
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        if not os.path.isdir(model_source):
            raise RuntimeError(f"Model dir {model_source} does not exist or is not a directory.")
    logging.info(f"Environment prepared. Model directory: {model_source}")


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info("Starting compositional generalization experiment")
    args = parse_args()
    logging.info(f"Arguments: model_dir={args.model_dir}, config={args.config}, degree={args.degree}")

    # Load configs
    logging.info(f"Loading configuration from {args.config}")
    configs = Config.load_config(args.config)
    logging.getLogger().setLevel(logging.DEBUG if configs.debug else logging.INFO)
    logging.info(f"Debug logging is {'enabled' if configs.debug else 'disabled'}")
    logging.info(f"Configuration loaded: num_shots={configs.num_shots}, alpha={configs.alpha}, block_idx={configs.block_idx}")

    prepare_environment(args.model_dir, local_files_only=True)

    # Create Random Function 
    logging.info("Generating two-hop function dataset")
    x1 = list(range(1, 4))
    x2 = list(range(1, 4))
    x3 = list(range(1, 4))

    intermediate = list(range(1, 4))
    outputs = list(range(1, 4))

    function = generate_two_hop_function(
        x1, x2, x3,
        intermediate,
        outputs,
        seed=42
    )
    logging.info(f"Generated function with {len(function)} demo pairs")

    logging.info(f"Building dataset with num_shots={configs.num_shots}")
    builder = DatasetBuilder(function, num_shots=configs.num_shots)
    
    dataset = builder.get_dataset(coverage_degree=args.degree)
    logging.info(f"Generated queries scanned: {builder.last_generated_count}")
    logging.info(f"Filtered dataset for coverage_degree={args.degree}: {len(dataset)} queries")

    # Create and run experiment
    logging.info("Initializing experiment...")
    experiment = TaskVectorExperiment(dataset=dataset, model_path=args.model_dir, configs=configs)
    logging.info("Running experiment...")
    experiment.run()
    logging.info("Experiment completed successfully")

if __name__ == "__main__":
    main()
