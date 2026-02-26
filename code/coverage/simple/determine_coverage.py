from __future__ import annotations

from collections import defaultdict
from heapq import heappop, heappush
from itertools import combinations
from typing import Dict, List, Set, Tuple

from dataset.demonstration_pair import DemoPair
from dataset.query import Query


Pair = Tuple[int, int]
Node = Tuple[Pair, int]          # ((x1,x2), x3)
TrainEx = Tuple[int, int, int, int]  # (x1,x2,x3,y)


def _pair_equiv_strength(train: List[TrainEx]) -> Dict[Tuple[Pair, Pair], int]:
    """
    Compute k(p,q): number of distinct x3 contexts where both pairs appear and have the same y.
    Strict version: if any shared context has mismatched y (or ambiguity), k(p,q)=0.
    Returns strengths for ordered keys (p,q).
    """
    # pair -> x3 -> set(y)
    pc_to_y: Dict[Pair, Dict[int, Set[int]]] = defaultdict(lambda: defaultdict(set))
    pairs: Set[Pair] = set()

    for x1, x2, x3, y in train:
        p = (x1, x2)
        pairs.add(p)
        pc_to_y[p][x3].add(y)

    strengths: Dict[Tuple[Pair, Pair], int] = {}

    plist = list(pairs)
    for p, q in combinations(plist, 2):
        pmap = pc_to_y[p]
        qmap = pc_to_y[q]
        common_ctx = set(pmap.keys()) & set(qmap.keys())
        if not common_ctx:
            continue

        k = 0
        ok = True
        for c in common_ctx:
            yp = pmap[c]
            yq = qmap[c]
            if len(yp) != 1 or len(yq) != 1:
                ok = False
                break
            if next(iter(yp)) != next(iter(yq)):
                ok = False
                break
            k += 1

        if ok and k > 0:
            strengths[(p, q)] = k
            strengths[(q, p)] = k

    return strengths


def max_coverage_degree_pair_substitution(train: List[TrainEx], test: Tuple[int, int, int]) -> int:
    """
    Nodes are (pair, x3). Training nodes are those appearing in train (ignoring y).
    Edge between (p, c) and (q, c) exists with weight k(p,q) if training supports p~q with strength k.
    This allows connecting test (p_test, c_test) to a training node (q, c_test) even if (p_test, c_test) wasn't seen.

    Returns k* = widest-path bottleneck from test node to any training node.
    """
    if not train:
        return 0

    strengths = _pair_equiv_strength(train)

    # Build the set of training nodes (pair, x3)
    train_nodes: Set[Node] = set()
    pairs_in_train: Set[Pair] = set()
    x3_values: Set[int] = set()

    for x1, x2, x3, _y in train:
        p = (x1, x2)
        pairs_in_train.add(p)
        x3_values.add(x3)
        train_nodes.add((p, x3))

    test_pair: Pair = (test[0], test[1])
    test_x3: int = test[2]
    test_node: Node = (test_pair, test_x3)

    # Finite node set we consider:
    # - all training nodes
    # - plus the test node
    # (We don't need to materialize nodes for unseen (pair, x3) unless they appear in train or are the test.)
    nodes: Set[Node] = set(train_nodes)
    nodes.add(test_node)

    # Adjacency: connect only nodes with SAME x3 via pair substitutions.
    # For a fixed x3=c, connect (p,c) to (q,c) if k(p,q)>0.
    adj: Dict[Node, List[Tuple[Node, int]]] = defaultdict(list)

    # Group training pairs by x3 that exist as nodes (so we can connect test to them too)
    pairs_by_x3: Dict[int, Set[Pair]] = defaultdict(set)
    for (p, c) in train_nodes:
        pairs_by_x3[c].add(p)
    # Also include test pair in its x3 bucket so we connect it as well
    pairs_by_x3[test_x3].add(test_pair)

    # Build edges per x3 layer
    for c, plist in pairs_by_x3.items():
        plist = list(plist)
        for i in range(len(plist)):
            for j in range(i + 1, len(plist)):
                p, q = plist[i], plist[j]
                w = strengths.get((p, q), 0)
                if w > 0:
                    u = (p, c)
                    v = (q, c)
                    # only add if nodes exist in our finite set
                    if u in nodes and v in nodes:
                        adj[u].append((v, w))
                        adj[v].append((u, w))

    # Widest path from test_node to any training node
    dist: Dict[Node, int] = {test_node: 10**18}
    pq: List[Tuple[int, Node]] = [(-dist[test_node], test_node)]
    visited: Set[Node] = set()

    while pq:
        neg_b, u = heappop(pq)
        b = -neg_b
        if u in visited:
            continue
        visited.add(u)

        if u in train_nodes:
            # first time we pop a train node is optimal widest-bottleneck
            return int(b if b != 10**18 else 0)

        for v, w in adj.get(u, []):
            cand = min(b, w)
            if cand > dist.get(v, 0):
                dist[v] = cand
                heappush(pq, (-cand, v))

    return 0

def convert_to_list(demo: DemoPair) -> List[TrainEx]:
    """Convert a DemoPair to a list of TrainEx tuples."""
    meta = demo.meta
    return [(meta['x1'], meta['x2'], meta['x3'], meta['y'])]

def max_coverage_degree_pair_substitution_wrapper(query: Query) -> int:
    """
    Wrapper to compute coverage degree for a Query instance.
    Expects query.query_demo.meta to contain 'x1', 'x2', 'x3' for the test example,
    and 'train_data' as a list of (x1,x2,x3,y) tuples for training examples.
    """
    demos = query.demonstrations
    train = []
    for d in demos:
        train.extend(convert_to_list(d))
    question = query.query_demo
    test = (question.meta.get('x1'), question.meta.get('x2'), question.meta.get('x3'))
    return max_coverage_degree_pair_substitution(train, test)
