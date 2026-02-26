import random
import json
import logging
from itertools import product
from typing import List

from dataset.demonstration_pair import DemoPair
from function.function import Function


def generate_two_hop_function(
    x1_range: List[int],
    x2_range: List[int],
    x3_range: List[int],
    intermediate_range: List[int],
    output_range: List[int],
    seed: int = None,
) -> Function:
    """
    Generates structured mapping:

        b = h(x1, x2)
        y = g(b, x3)

    Returns a Function object that contains unique-input DemoPair objects.
    """

    if seed is not None:
        random.seed(seed)
        logging.debug(f"Random seed set to {seed}")

    x1_domain = list(x1_range)
    x2_domain = list(x2_range)
    x3_domain = list(x3_range)
    intermediate_domain = list(intermediate_range)
    output_domain = list(output_range)
    
    logging.debug(f"Domain sizes: x1={len(x1_domain)}, x2={len(x2_domain)}, x3={len(x3_domain)}, intermediate={len(intermediate_domain)}, output={len(output_domain)}")

    # First hop: h(x1, x2) -> intermediate
    logging.debug("Building first hop mapping h(x1, x2) -> intermediate")
    h = {}
    for x1, x2 in product(x1_domain, x2_domain):
        h[(x1, x2)] = random.choice(intermediate_domain)
    logging.debug(f"First hop mapping created with {len(h)} entries")

    # Second hop: g(b, x3) -> output
    logging.debug("Building second hop mapping g(b, x3) -> output")
    g = {}
    for b, x3 in product(intermediate_domain, x3_domain):
        g[(b, x3)] = random.choice(output_domain)
    logging.debug(f"Second hop mapping created with {len(g)} entries")

    logging.debug("Creating demo pairs from two-hop composition")
    function_obj = Function()
    for idx, (x1, x2, x3) in enumerate(product(x1_domain, x2_domain, x3_domain)):
        b = h[(x1, x2)]
        y = g[(b, x3)]
        function_obj.add_demo_pair(
            DemoPair(
                input=f"({x1},{x2},{x3})",
                output=str(y),
                id=idx,
                meta={
                    "x1": x1,
                    "x2": x2,
                    "x3": x3,
                    "intermediate": b,
                },
            )
        )
    
    logging.debug(f"Created function with {len(function_obj)} demo pairs")
    return function_obj
