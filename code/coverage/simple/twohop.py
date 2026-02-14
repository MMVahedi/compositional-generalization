import random
import json
from itertools import product
from typing import List, Dict, Tuple


def generate_two_hop_dataset(
    x1_domain: List[int],
    x2_domain: List[int],
    x3_domain: List[int],
    intermediate_domain: List[int],
    output_domain: List[int],
    seed: int = None,
) -> Dict:
    """
    Generates structured mapping:

        b = h(x1, x2)
        y = g(b, x3)

    Returns dictionary containing:
        - full dataset
        - h mapping
        - g mapping
    """

    if seed is not None:
        random.seed(seed)

    # First hop: h(x1, x2) -> intermediate
    h = {}
    for x1, x2 in product(x1_domain, x2_domain):
        h[(x1, x2)] = random.choice(intermediate_domain)

    # Second hop: g(b, x3) -> output
    g = {}
    for b, x3 in product(intermediate_domain, x3_domain):
        g[(b, x3)] = random.choice(output_domain)

    dataset = []
    for idx, (x1, x2, x3) in enumerate(product(x1_domain, x2_domain, x3_domain)):
        b = h[(x1, x2)]
        y = g[(b, x3)]
        # Format as DemoPair-compatible object
        dataset.append({
            "input": f"({x1},{x2},{x3})",
            "output": str(y),
            "id": idx,
            "meta": {
                "x1": x1,
                "x2": x2,
                "x3": x3,
                "intermediate": b
            }
        })

    return {
        "dataset": dataset,
        "h_mapping": {f"{k[0]},{k[1]}": v for k, v in h.items()},
        "g_mapping": {f"{k[0]},{k[1]}": v for k, v in g.items()}
    }


def save_dataset_to_json(data: Dict, filename: str):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


# ---------------- Example usage ----------------

if __name__ == "__main__":
    x1 = list(range(1, 10))
    x2 = list(range(1, 10))
    x3 = list(range(1, 10))

    intermediate = list(range(1, 10))
    outputs = list(range(1, 10))

    data = generate_two_hop_dataset(
        x1, x2, x3,
        intermediate,
        outputs,
        seed=42
    )

    save_dataset_to_json(data, "two_hop_dataset.json")

    print("Dataset saved to two_hop_dataset.json")
