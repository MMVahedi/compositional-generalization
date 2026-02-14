import argparse
import numpy as np
import random
from collections import defaultdict
import logging
import itertools
from typing import List
import multiprocessing as mp
from tqdm import tqdm
import os
import json
from functools import partial


def setup_logging(debug_mode: bool):
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s - %(message)s")


def make_arbitrary_function(domain: List[int], codomain: List[int], seen_ratio: float):
    result = {}
    range_vals = random.choices(codomain, k=len(domain))
    for operand, res in zip(domain, range_vals):
        result[operand] = res

    keys = list(result.keys())
    random.shuffle(keys)
    cutoff = int(round(seen_ratio * len(keys)))
    seen_expected_inputs = keys[:cutoff]

    S_dict = {inp: result[inp] for inp in seen_expected_inputs}
    return result, S_dict


# ---------- CHANGED ----------
def form_item(input_tokens: List[str], output_token: str):
    inp = ",".join(input_tokens)
    return {"input": inp, "output": output_token}


# ---------- CHANGED ----------
def form_item_2hop_target(h1_idx, h2_idx, h3_idx, t_idx,
                          vocab, f1_dict,
                          cot=False,
                          fake_bridge=False):
    input_tokens = [str(h1_idx), str(h2_idx), str(h3_idx)]
    inp_str = ",".join(input_tokens)

    # final output is ONLY the target token
    out_str = str(t_idx)

    return {"input": inp_str, "output": out_str}


def process_item_S_f1(item, S_f2_index):
    (h1, h2), b1 = item
    partial_set = set()
    if b1 not in S_f2_index:
        return partial_set
    for h3, t in S_f2_index[b1]:
        partial_set.add((h1, h2, h3, t))
    return partial_set


def coverage_type(sc1: bool, sc2: bool):
    bit = (sc1 << 1) + sc2
    return 0 if bit == 3 else bit + 1


def reservoir_update(reservoir, tup, total_count, capacity):
    if len(reservoir) < capacity:
        reservoir.append(form_item(tup[0], tup[1]))
    else:
        r = random.randint(0, total_count - 1)
        if r < capacity:
            reservoir[r] = form_item(tup[0], tup[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tokens", type=int, required=True)
    parser.add_argument("--same_f12", action="store_true")
    parser.add_argument("--default_seen_ratio", type=float, default=0.7)
    parser.add_argument("--max_data_num", type=int, default=382000)
    parser.add_argument("--coverage_reservoir_size", type=int, default=2000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--fake_bridge", action="store_true")
    args = parser.parse_args()

    setup_logging(args.debug)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ---------- CHANGED ----------
    vocab = [str(i) for i in range(args.num_tokens)]

    domain_f1 = list(itertools.product(range(args.num_tokens), range(args.num_tokens)))
    domain_f2 = list(itertools.product(range(args.num_tokens), range(args.num_tokens)))
    codomain = list(range(args.num_tokens))

    f1_dict, S_f1 = make_arbitrary_function(domain_f1, codomain, args.default_seen_ratio)
    if args.same_f12:
        f2_dict = dict(f1_dict)
        S_f2 = dict(S_f1)
    else:
        f2_dict, S_f2 = make_arbitrary_function(domain_f2, codomain, args.default_seen_ratio)

    S_f2_index = defaultdict(list)
    for (b1, h3), t in S_f2.items():
        S_f2_index[b1].append((h3, t))

    with mp.Pool(processes=round(mp.cpu_count() * 0.9)) as pool:
        process_f1 = partial(process_item_S_f1, S_f2_index=S_f2_index)
        inferred = list(tqdm(pool.imap(process_f1, S_f1.items()),
                             total=len(S_f1)))

    inferred_idx = set().union(*inferred)
    inferred_idx = set(random.sample(list(inferred_idx), args.max_data_num))

    all_data = []

    for (h1, h2, h3, t) in inferred_idx:
        item = form_item_2hop_target(
            h1, h2, h3, t,
            vocab=vocab,
            f1_dict=S_f1,
            cot=(not args.fake_bridge and args.cot),
            fake_bridge=args.fake_bridge
        )
        item["source"] = "inferred"
        all_data.append(item)

    f2_index = defaultdict(list)
    for (b1, h3), t in f2_dict.items():
        f2_index[b1].append((h3, t))

    coverage_reservoirs = defaultdict(list)
    coverage_seen_count = defaultdict(int)

    for (h1, h2), b1 in f1_dict.items():
        sc1 = (h1, h2) in S_f1
        for (h3, t) in f2_index[b1]:
            sc2 = (b1, h3) in S_f2
            ctype = coverage_type(sc1, sc2)
            coverage_seen_count[ctype] += 1
            reservoir_update(
                coverage_reservoirs[ctype],
                ([str(h1), str(h2), str(h3)], str(t)),
                coverage_seen_count[ctype],
                args.coverage_reservoir_size
            )

    for ctype, items in coverage_reservoirs.items():
        for it in items:
            it["source"] = f"coverage_{ctype}"
            all_data.append(it)

    atomic_facts_f1 = [
        form_item([str(h1), str(h2)], str(b1))
        for (h1, h2), b1 in f1_dict.items()
    ]

    atomic_facts_f2 = [
        form_item([str(b1), str(h3)], str(t))
        for (b1, h3), t in f2_dict.items()
    ]

    for idx, item in enumerate(all_data):
        item["id"] = idx

    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(
        base_dir,
        "data",
        f"twohop.{args.num_tokens}.{args.max_data_num}.{'same-f12' if args.same_f12 else 'diff-f12'}"
    )
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)
    with open(os.path.join(save_dir, "all_data.json"), "w") as f:
        json.dump(all_data, f, indent=2)
    with open(os.path.join(save_dir, "atomic_facts_f1.json"), "w") as f:
        json.dump(atomic_facts_f1, f, indent=2)
    with open(os.path.join(save_dir, "atomic_facts_f2.json"), "w") as f:
        json.dump(atomic_facts_f2, f, indent=2)

    print("[INFO] Done!")
    print(f"Total examples: {len(all_data)}")


if __name__ == "__main__":
    main()
