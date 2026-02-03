import json

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