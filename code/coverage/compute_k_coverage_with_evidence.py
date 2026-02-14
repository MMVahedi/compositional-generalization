from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple, FrozenSet, Set

import networkx as nx

# -------------------------
# Helpers
# -------------------------
def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_input_tokens(s: str) -> Tuple[int, ...]:
    """Parse strings like '9,4,9' into tuple of ints (9,4,9)."""
    return tuple(int(x) for x in s.split(","))


def powerset_indices(length: int) -> List[FrozenSet[int]]:
    """Return all non-empty, proper subsets of indices {0,...,length-1} as frozensets."""
    indices = list(range(length))
    subsets = []
    for r in range(1, length):
        for comb in combinations(indices, r):
            subsets.append(frozenset(comb))
    return subsets


# -------------------------
# Union-Find
# -------------------------
class UnionFind:
    def __init__(self):
        self.parent = {}
        self.size = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size.get(ra, 1) < self.size.get(rb, 1):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] = self.size.get(ra, 1) + self.size.get(rb, 1)


# -------------------------
# Core: functional-k equivalence
# -------------------------
def extract_subsequence(seq: Tuple[int, ...], indices: FrozenSet[int]) -> Tuple[int, ...]:
    return tuple(seq[i] for i in sorted(indices))


def build_equiv_classes_for_subset(
    dataset_map: Dict[Tuple[int, ...], int],
    subset_indices: FrozenSet[int],
    min_evidence: int = 1,
) -> UnionFind:
    logging.debug(
        f"Building equivalence classes for subset indices={sorted(subset_indices)} (k={min_evidence})"
    )

    complement_indices = frozenset(range(len(next(iter(dataset_map))))) - subset_indices
    behavior: Dict[Tuple[int, ...], Dict[Tuple[int, ...], int]] = defaultdict(dict)

    for full_seq, label in dataset_map.items():
        subseq = extract_subsequence(full_seq, subset_indices)
        complement = extract_subsequence(full_seq, complement_indices)
        if complement in behavior[subseq] and behavior[subseq][complement] != label:
            logging.debug(
                f"Contradiction for exact input {subseq}+{complement}: "
                f"{behavior[subseq][complement]} vs {label}"
            )
        behavior[subseq][complement] = label

    uf = UnionFind()
    subseqs = list(behavior.keys())
    comp_sets = {s: set(behavior[s].keys()) for s in subseqs}

    for i, s1 in enumerate(subseqs):
        for s2 in subseqs[i + 1 :]:
            shared_complements = comp_sets[s1] & comp_sets[s2]
            if not shared_complements:
                continue

            matching_count = 0
            contradiction_found = False
            for c in shared_complements:
                if behavior[s1][c] != behavior[s2][c]:
                    contradiction_found = True
                    break
                matching_count += 1

            if not contradiction_found and matching_count >= min_evidence:
                uf.union(s1, s2)

    for s in subseqs:
        uf.find(s)

    num_classes = len({uf.find(s) for s in subseqs})
    logging.info(
        f"Subset {sorted(subset_indices)}: {num_classes} equiv classes from {len(subseqs)} subsequences"
    )
    return uf


def build_all_equiv_classes(
    dataset_map: Dict[Tuple[int, ...], int],
    length: int,
    min_evidence: int = 1,
) -> Dict[FrozenSet[int], UnionFind]:
    subsets = powerset_indices(length)
    return {
        subset: build_equiv_classes_for_subset(dataset_map, subset, min_evidence)
        for subset in subsets
    }


# -------------------------
# Build substitution graph + grouped evidence (IDs)
# -------------------------
def reconstruct_full_from(subseq: Tuple[int, ...], complement: Tuple[int, ...], subset_indices: FrozenSet[int], length: int) -> Tuple[int, ...]:
    """Reconstruct full sequence (tuple) given subsequence at subset_indices and complement at others."""
    full = [None] * length
    subset_sorted = sorted(subset_indices)
    complement_indices = sorted(set(range(length)) - set(subset_indices))
    for i, idx in enumerate(subset_sorted):
        full[idx] = subseq[i]
    for j, idx in enumerate(complement_indices):
        full[idx] = complement[j]
    return tuple(full)


def build_subst_graph_with_grouped_evidence(
    nodes: List[Tuple[int, ...]],
    triple2t: Dict[Tuple[int, ...], int],
    behavior_maps: Dict[FrozenSet[int], Dict[Tuple[int, ...], Dict[Tuple[int, ...], int]]],
    equiv_classes: Dict[FrozenSet[int], UnionFind],
    min_evidence: int,
    tr2id: Dict[Tuple[int, ...], int],
):
    """
    Build substitution graph and collect grouped evidence for each subsequence pair (a,a') under subset I.
    example_pairs are stored as pairs of IDs [id_a, id_b] rather than full sequences.

    Returns:
      - G: graph
      - grouped_evidence: list of entries (one per (I,a,a')) with aggregated evidence

    Note about `shared_complements`:
      - Each complement is the tuple of values at indices J = [0..ℓ-1]\I (the 'context' outside subsequence I).
      - `shared_complements` are the specific contexts (values at indices J) that appear in the dataset
        for both subsequences a and a' and for which the labels match. These are the co-occurrences
        that provide the k evidence for functional equivalence under subset I.
    """
    G = nx.Graph()
    G.add_nodes_from(nodes)
    length = len(nodes[0])

    # key: (subset_tuple, subseq_a, subseq_b) where subseq_a <= subseq_b lexicographically
    grouped: Dict[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]], Dict] = {}

    for subset_indices, uf in equiv_classes.items():
        subset_tuple = tuple(sorted(subset_indices))
        complement_indices = sorted(set(range(length)) - set(subset_indices))

        # bucket by complement
        buckets: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = defaultdict(list)
        for n in nodes:
            comp = tuple(n[i] for i in complement_indices)
            buckets[comp].append(n)

        for comp, bucket_nodes in buckets.items():
            if len(bucket_nodes) < 2:
                continue

            # group by UF representative of subseq
            group_by_repr: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = defaultdict(list)
            for n in bucket_nodes:
                subseq = tuple(n[i] for i in sorted(subset_indices))
                rep = uf.find(subseq)
                group_by_repr[rep].append(n)

            for group in group_by_repr.values():
                if len(group) < 2:
                    continue
                # pairwise within group
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        a = group[i]
                        b = group[j]
                        # defensive label check
                        if a in triple2t and b in triple2t and triple2t[a] != triple2t[b]:
                            continue

                        subseq_a = tuple(a[idx] for idx in sorted(subset_indices))
                        subseq_b = tuple(b[idx] for idx in sorted(subset_indices))

                        # >>> skip identical subsequence pairs (a == a') — avoids trivial "2","2" entries
                        if subseq_a == subseq_b:
                            continue

                        behavior = behavior_maps.get(subset_indices, {})
                        comps_a = set(behavior.get(subseq_a, {}).keys())
                        comps_b = set(behavior.get(subseq_b, {}).keys())
                        shared = comps_a & comps_b

                        # filter only matches where labels equal
                        shared_matched = [c for c in shared if behavior[subseq_a][c] == behavior[subseq_b][c]]

                        if not shared_matched:
                            continue

                        # We'll only register a matched complement as evidence if it yields
                        # a real *distinct* pair of dataset ids (id_a != id_b).
                        # This ensures example_pair_ids is the true witness set.
                        # canonicalize subseq pair ordering to avoid duplicates
                        if subseq_a <= subseq_b:
                            key = (subset_tuple, subseq_a, subseq_b)
                            flip = False
                        else:
                            key = (subset_tuple, subseq_b, subseq_a)
                            flip = True

                        entry = grouped.setdefault(key, {
                            "subset": list(subset_tuple),
                            # subseq_pair stored as two strings (e.g. "0,1")
                            "subseq_pair": [",".join(map(str, key[1])), ",".join(map(str, key[2]))],
                            "shared_complements_set": set(),  # temp: set of complement tuples actually realized by id pairs
                            "example_pair_ids_set": set(),  # temp: set of id-pair strings for uniqueness
                            "example_pair_ids": [],  # final: list of [id_a, id_b]
                        })

                        # For each matched complement try to produce distinct ids; only keep those that succeed
                        for c in shared_matched:
                            # reconstruct original order of full pair consistent with subseq ordering
                            if not flip:
                                full_a = reconstruct_full_from(subseq_a, c, subset_indices, length)
                                full_b = reconstruct_full_from(subseq_b, c, subset_indices, length)
                            else:
                                full_a = reconstruct_full_from(subseq_b, c, subset_indices, length)
                                full_b = reconstruct_full_from(subseq_a, c, subset_indices, length)

                            # map to ids (must exist in tr2id)
                            id_a = tr2id.get(full_a)
                            id_b = tr2id.get(full_b)
                            if id_a is None or id_b is None:
                                # skip if either full example not in dataset
                                continue

                            # Prevent self-pairs (id_a == id_b)
                            if id_a == id_b:
                                continue

                            pair_key = f"{id_a}|{id_b}"
                            if pair_key not in entry["example_pair_ids_set"]:
                                entry["example_pair_ids_set"].add(pair_key)
                                entry["example_pair_ids"].append([id_a, id_b])
                                # only add complement if we successfully produced an id-pair witness
                                entry["shared_complements_set"].add(tuple(c))

                        # If this grouped entry has at least min_evidence *real* witnesses, add edge
                        if len(entry["shared_complements_set"]) >= min_evidence:
                            G.add_edge(a, b)
                        else:
                            # if we did not get enough **real** complements for this pair, do not keep the entry
                            # (we'll drop it later in finalization by virtue of example_pair_ids being empty)
                            pass

    # finalize grouped entries into a JSON-serializable list
    grouped_list = []
    for key, entry in grouped.items():
        # only output entries that actually have example_pair_ids (i.e., produced real witnesses)
        if not entry["example_pair_ids"]:
            continue
        shared_complements = [list(c) for c in sorted(entry["shared_complements_set"])]
        grouped_list.append({
            "subset": entry["subset"],
            # subseq_pair is array of two strings
            "subseq_pair": entry["subseq_pair"],
            "matching_count": len(shared_complements),
            "shared_complements": shared_complements,
            "example_pair_ids": entry["example_pair_ids"],  # list of [id_a, id_b]
        })

    logging.info(f"Built subst graph with grouped evidence: |V|={len(nodes)}, |E|={G.number_of_edges()} grouped_entries={len(grouped_list)}")
    return G, grouped_list


# -------------------------
# k-coverage computation (single k)
# -------------------------
def compute_k_coverage_for_dataset_single(
    parsed: List[Tuple[Tuple[int, ...], int]],
    all_items: List[dict],
    triple2t: Dict[Tuple[int, ...], int],
    min_evidence: int,
):
    """
    Compute substitution graph (for given k) and return:
      - G (graph)
      - covered_nodes (set of nodes connected to at least one observed example)
      - comp_map (node -> list of ids in same component)
      - grouped evidence list (with example_pair_ids)
    """

    # dataset_map: only sequences with labels (required for behavior checks)
    if triple2t:
        dataset_map = {tr: triple2t[tr] for tr in triple2t}
    else:
        # if no labels present, fallback to presence-only
        dataset_map = {tr: 0 for tr, _ in parsed}

    length = len(parsed[0][0])
    equiv_classes = build_all_equiv_classes(dataset_map, length, min_evidence)

    # build behavior maps for evidence (using dataset_map)
    behavior_maps = {}
    subsets = powerset_indices(length)
    for subset in subsets:
        complement_indices = sorted(set(range(length)) - set(subset))
        behavior = defaultdict(dict)
        for full_seq, label in dataset_map.items():
            subseq = tuple(full_seq[i] for i in sorted(subset))
            complement = tuple(full_seq[i] for i in complement_indices)
            behavior[subseq][complement] = label
        behavior_maps[subset] = behavior

    nodes = [tr for tr, _ in parsed]
    tr2id = {tr: idx for tr, idx in parsed}
    G, grouped_evidence = build_subst_graph_with_grouped_evidence(nodes, triple2t, behavior_maps, equiv_classes, min_evidence, tr2id)

    # identify observed/training nodes: items with source == "inferred"
    observed_nodes = set()
    for it in all_items:
        if it.get("source") == "inferred":
            tr = parse_input_tokens(it["input"])
            observed_nodes.add(tr)

    # compute components and coverage
    comp_map = {}
    covered_nodes = set()
    for comp in nx.connected_components(G):
        comp_ids = [tr2id[tr] for tr in comp]
        # if any observed node in component -> entire comp is covered
        if any(tr in observed_nodes for tr in comp):
            covered_nodes.update(comp)
        for tr in comp:
            comp_map[tr] = comp_ids

    return G, covered_nodes, comp_map, grouped_evidence


# -------------------------
# k-sweep orchestration
# -------------------------
def run_k_sweep(all_items: List[dict], k_min: int, k_max: int, data_dir: str, debug: bool = False):
    # parse dataset once
    parsed = []
    triple2t: Dict[Tuple[int, ...], int] = {}
    for it in all_items:
        tr = parse_input_tokens(it["input"])
        parsed.append((tr, it["id"]))
        if "output" in it and it["output"] != "":
            try:
                triple2t[tr] = int(it["output"])
            except Exception:
                pass

    total_items = len(all_items)
    # Track minimal k where each item becomes covered (0 => never covered)
    min_k_by_id = {it["id"]: 0 for it in all_items}

    k_results = {}

    for k in range(k_min, k_max + 1):
        logging.info(f"Running k-sweep for k={k} ...")
        G, covered_nodes, comp_map, grouped_evidence = compute_k_coverage_for_dataset_single(parsed, all_items, triple2t, k)

        # Save grouped evidence for this k
        edge_file = os.path.join(data_dir, f"grouped_edge_evidence_k{k}.json")
        with open(edge_file, "w", encoding="utf-8") as f:
            json.dump(grouped_evidence, f, indent=2)
        logging.info(f"Wrote grouped edge evidence for k={k} to {edge_file}")

        # Map covered nodes to ids
        covered_ids = set()
        for tr in covered_nodes:
            # every tr must be in parsed -> find id
            for p_tr, p_id in parsed:
                if p_tr == tr:
                    covered_ids.add(p_id)
                    break

        # Update minimal k for newly covered items
        newly_covered = 0
        for cid in covered_ids:
            if min_k_by_id[cid] == 0:
                min_k_by_id[cid] = k
                newly_covered += 1

        # Compute per-source coverage breakdown
        source_totals = defaultdict(int)
        source_covered = defaultdict(int)
        for it in all_items:
            source_totals[it.get("source", "UNK")] += 1
            if it["id"] in covered_ids:
                source_covered[it.get("source", "UNK")] += 1

        # Save summary for this k
        k_results[k] = {
            "k": k,
            "total_items": total_items,
            "total_covered": len(covered_ids),
            "newly_covered": newly_covered,
            "coverage_pct": 100.0 * len(covered_ids) / total_items if total_items > 0 else 0.0,
            "source_totals": dict(source_totals),
            "source_covered": dict(source_covered),
            "edge_file": os.path.basename(edge_file),
        }

        logging.info(
            f"k={k}: covered {len(covered_ids)}/{total_items} ({k_results[k]['coverage_pct']:.2f}%), newly covered={newly_covered}"
        )

    # After sweep, annotate each item with minimal k (0 if never covered)
    annotated = []
    for it in all_items:
        new_it = dict(it)
        new_it["min_k_covered"] = min_k_by_id[it["id"]]
        annotated.append(new_it)

    return k_results, annotated


# -------------------------
# Main CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--k_sweep", action="store_true")
    ap.add_argument("--k_min", type=int, default=1)
    ap.add_argument("--k_max", type=int, default=20)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--out_name", type=str, default=None)
    args = ap.parse_args()

    setup_logging(args.debug)

    with open(os.path.join(args.data_dir, "all_data.json"), "r", encoding="utf-8") as f:
        all_items = json.load(f)

    existing_ids = {it.get("id") for it in all_items}
    if None in existing_ids or len(existing_ids) != len(all_items):
        for i, it in enumerate(all_items):
            it["id"] = i

    if args.k_sweep:
        logging.info(f"Running k-sweep from k={args.k_min} to k={args.k_max}")
        k_results, annotated = run_k_sweep(all_items, args.k_min, args.k_max, args.data_dir, debug=args.debug)

        # Write results
        os.makedirs(args.data_dir, exist_ok=True)
        k_results_path = os.path.join(args.data_dir, "k_sweep_results.json")
        with open(k_results_path, "w", encoding="utf-8") as f:
            json.dump(k_results, f, indent=2)

        out_name = args.out_name or f"all_data_annotated_k_sweep.json"
        out_path = os.path.join(args.data_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(annotated, f, indent=2)

        logging.info(f"K-sweep results saved to {k_results_path}")
        logging.info(f"Annotated dataset with 'min_k_covered' saved to {out_path}")
    else:
        # Single-k mode
        parsed = [(parse_input_tokens(it["input"]), it["id"]) for it in all_items]
        triple2t = {tuple(parse_input_tokens(it["input"])): int(it["output"]) for it in all_items if "output" in it and it["output"] != ""}

        G, covered_nodes, comp_map, grouped_evidence = compute_k_coverage_for_dataset_single(parsed, all_items, triple2t, args.k)

        # Save grouped evidence
        edge_file = os.path.join(args.data_dir, f"grouped_edge_evidence_k{args.k}.json")
        with open(edge_file, "w", encoding="utf-8") as f:
            json.dump(grouped_evidence, f, indent=2)
        logging.info(f"Wrote grouped edge evidence to {edge_file}")

        # Build annotated similar to previous function
        annotated_items = []
        # need tr->id map to find comp ids
        tr2id = {tr: idx for tr, idx in parsed}
        for it in all_items:
            tr = parse_input_tokens(it["input"])
            comp_ids = comp_map.get(tr, [it["id"]])
            new_it = dict(it)
            new_it["k_component_ids"] = sorted(comp_ids)
            annotated_items.append(new_it)

        out_name = args.out_name or f"all_data_annotated_k{args.k}.json"
        out_path = os.path.join(args.data_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(annotated_items, f, indent=2)

        logging.info(f"Annotated dataset written to: {out_path}")
        logging.info(f"Each item now contains 'k_component_ids' (k={args.k}).")


if __name__ == "__main__":
    main()
