import json
from typing import List, Sequence

from coverage.demonstration_pair import DemoPair
from task_vector_pkg.query import Query


def get_demo_pairs(demos_path: str) -> List[DemoPair]:
    """Load demo pairs from a JSON file."""
    with open(demos_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data["dataset"]
    for item in items:
        if isinstance(item, dict):
            DemoPair.from_dict(item)
        else:
            raise ValueError(f"Unsupported demo entry format: {item}. Expected dict with 'input', 'output', 'id', 'meta'.")
