import json
import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Union
import torch

class Config:
    def __init__(self, max_tokens, block_idx, alpha, average_separators, normalize, system_prompt, temperature, top_k, top_p, num_shots, sep, debug=False):
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
        self.debug = debug

    @staticmethod
    def load_config(config_path: str) -> "Config":
        logging.debug(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        logging.debug(f"Config keys found: {list(config.keys())}")
        return Config(
            max_tokens=config['max_tokens'],
            block_idx=config['block_idx'],
            alpha=config['alpha'],
            average_separators=config['average_separators'],
            normalize=config['normalize'],
            system_prompt=config['system_prompt'],
            temperature=config['temperature'],
            top_k=config['top_k'],
            top_p=config['top_p'],
            num_shots=config.get('num_shots'),
            sep=config.get('sep'),
            debug=config.get('debug', False),
        )

@dataclass
class TaskVectorConfig:
    # extraction
    # layer_idx may be a single int or a sequence of ints to extract from multiple layers
    layer_idx: Union[int, Sequence[int]] = -1                   # which hidden_states index to read (supports negative)
    average_separators: bool = False      # False => last separator only; True => mean over all
    normalize: Optional[str] = None       # None | "l2"

    # injection
    alpha: float = 1.0
    device: Union[str, torch.device] = "cpu"


@dataclass
class TaskVector:
    vector: torch.Tensor                  # [d_model] or [n_layers, d_model]
    # layer_idx may be int or sequence[int]
    layer_idx: Union[int, Sequence[int]]
    separator_text: str
    average_separators: bool
    meta: dict
