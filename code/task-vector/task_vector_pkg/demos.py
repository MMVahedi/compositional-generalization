import os
import json
from typing import List, Sequence

from coverage.demontration_pair import DemoPair
from task_vector_pkg.query import Query


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


def group_demos(demos: Sequence[DemoPair], group_size: int, tokenizer=None):
    """Yield Query objects, each with (group_size-1) shots + 1 query."""
    n = len(demos) // group_size
    for i in range(n):
        start = i * group_size
        shots = demos[start : start + (group_size - 1)]
        query_demo = demos[start + (group_size - 1)]
        yield Query(shots, query_demo)