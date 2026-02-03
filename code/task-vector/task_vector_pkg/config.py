import json


class Config:
    def __init__(self, max_tokens, block_idx, alpha, average_separators, normalize, system_prompt, temperature, top_k, top_p, num_shots, sep):
        self.max_tokens = max_tokens
        self.block_idx = block_idx
        self.alpha = alpha
        self.average_separators = average_separators
        self.normalize = normalize
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.num_shots = num_shots
        self.sep = sep


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config
