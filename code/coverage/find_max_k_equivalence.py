from __future__ import annotations
import itertools
from typing import List, Dict, Tuple, FrozenSet, Any


def parse_input_tokens(s: Any) -> Tuple[int, ...]:
    """Supports either tuple/list of ints or '1,2,3' strings."""
    if isinstance(s, (tuple, list)):
        return tuple(int(x) for x in s)
    if isinstance(s, str):
        s = s.strip()
        if s == "":
            return tuple()
        return tuple(int(x) for x in s.split(","))
    raise ValueError("input must be tuple/list or comma-separated string")


def powerset_indices(length: int):
    indices = list(range(length))
    subsets = []
    for r in range(1, length):
        for comb in itertools.combinations(indices, r):
            subsets.append(frozenset(comb))
    return subsets


def find_max_k_equivalence(context: List[Dict], verbose: bool = False):
    """
    context: list of dicts each containing at least:
        - "input": "i0,i1,...,in-1"  (or tuple/list of ints)
        - "output": any hashable label
      optionally:
        - "id": int (if absent we'll assign 0..N-1)
    Returns:
      {
        "max_k": int,
        "matches": [ { "I": [indices], "a": "...,...", "a_prime": "...,...",
                       "matching_count": k,
                       "shared_complements": [ [c1,...], ...],
                       "example_pair_ids": [ [id_x, id_xprime], ...]
                      }, ... ]
      }
    """
    # Normalize dataset and ensure ids
    parsed = []
    for idx, it in enumerate(context):
        tid = it.get("id", idx)
        tr = parse_input_tokens(it["input"])
        out = it.get("output", None)
        parsed.append((tr, out, tid))

    if not parsed:
        return {"max_k": 0, "matches": []}

    n = len(parsed[0][0])
    # build quick lookup: for each full seq -> (output, id)
    # but we also need behavior maps per subset I
    max_k = 0
    matches = []

    subsets = powerset_indices(n)

    for I in subsets:
        # complement indices J
        J = tuple(sorted(set(range(n)) - set(I)))

        # behavior: subseq -> { complement_tuple -> (output, id) }
        behavior: Dict[Tuple[int, ...], Dict[Tuple[int, ...], Tuple[Any, int]]] = {}

        for full, out, tid in parsed:
            subseq = tuple(full[i] for i in sorted(I))
            complement = tuple(full[j] for j in J)
            behavior.setdefault(subseq, {})
            # if same subseq+complement appears multiple times with conflicting outputs,
            # we still record the first — but consistency is checked later across subseqs
            behavior[subseq][complement] = (out, tid)

        subseqs = list(behavior.keys())
        # examine every unordered pair a,a'
        for i in range(len(subseqs)):
            for j in range(i + 1, len(subseqs)):
                a = subseqs[i]
                a_p = subseqs[j]

                # Find complements that appear for both a and a'
                comps_a = set(behavior[a].keys())
                comps_b = set(behavior[a_p].keys())
                shared_complements = comps_a & comps_b
                if not shared_complements:
                    continue

                # Consistency check: *every* shared complement that appears in D must have equal outputs
                inconsistent = False
                for c in shared_complements:
                    out1, _ = behavior[a][c]
                    out2, _ = behavior[a_p][c]
                    if out1 != out2:
                        inconsistent = True
                        break
                if inconsistent:
                    # fails Definition 3.1 consistency, cannot be equivalent for any k
                    continue

                # Now collect only *distinct* I-co-occurrence witnesses:
                # we require pairs of distinct dataset items (id_x != id_x')
                witness_pairs = []
                witness_complements = []
                for c in sorted(shared_complements):
                    out1, id1 = behavior[a][c]
                    out2, id2 = behavior[a_p][c]
                    if id1 == id2:
                        # same example cannot form an I-co-occurrence
                        continue
                    # record witness (id pair) and complement
                    witness_pairs.append([id1, id2])
                    witness_complements.append(list(c))

                k_here = len(witness_complements)
                if k_here == 0:
                    continue

                # This pair (a,a') is functionally k-equivalent for any k <= k_here.
                # We are looking for max overall, so compare:
                if k_here > max_k:
                    max_k = k_here
                    matches = [
                        {
                            "I": sorted(list(I)),
                            "a": ",".join(map(str, a)),
                            "a_prime": ",".join(map(str, a_p)),
                            "matching_count": k_here,
                            "shared_complements": witness_complements,
                            "example_pair_ids": witness_pairs,
                        }
                    ]
                elif k_here == max_k:
                    matches.append(
                        {
                            "I": sorted(list(I)),
                            "a": ",".join(map(str, a)),
                            "a_prime": ",".join(map(str, a_p)),
                            "matching_count": k_here,
                            "shared_complements": witness_complements,
                            "example_pair_ids": witness_pairs,
                        }
                    )

                if verbose:
                    print(f"I={sorted(list(I))}, a={a}, a'={a_p}, k_here={k_here}, witnesses={witness_pairs}")

    return {"max_k": max_k, "matches": matches}


def find_k_equivalences_for_query(
    context: List[Dict],
    query: Dict,
    verbose: bool = False,
) -> List[Dict]:
    """
    Given a context D and a single query input, return every functional
    k-equivalence (I, a, a') witnessed in D such that the query participates —
    i.e. query_I == a  OR  query_I == a'.

    Parameters
    ----------
    context : list of dicts, each with "input" (and optionally "output", "id")
    query   : dict with at least "input" (output/id are ignored)

    Returns
    -------
    List of dicts, one per relevant equivalence:
      {
        "I"                 : [sorted index list],
        "query_subseq"      : "v0,v1,..."  (the side the query matches),
        "equivalent_subseq" : "v0,v1,..."  (the other side),
        "k"                 : int,          (number of co-occurrence witnesses),
        "shared_complements": [[c, ...], ...],
        "example_pair_ids"  : [[id_x, id_xp], ...],
      }
    """
    # ---- parse context ----
    parsed = []
    for idx, it in enumerate(context):
        tid = it.get("id", idx)
        tr = parse_input_tokens(it["input"])
        out = it.get("output", None)
        parsed.append((tr, out, tid))

    if not parsed:
        return []

    n = len(parsed[0][0])

    # ---- parse query ----
    query_tokens = parse_input_tokens(query["input"])
    if len(query_tokens) != n:
        raise ValueError(
            f"Query length {len(query_tokens)} != context input length {n}"
        )

    results: List[Dict] = []
    subsets = powerset_indices(n)

    for I in subsets:
        J = tuple(sorted(set(range(n)) - set(I)))

        # build behavior map:  subseq -> { complement -> (output, id) }
        behavior: Dict[Tuple[int, ...], Dict[Tuple[int, ...], Tuple[Any, int]]] = {}
        for full, out, tid in parsed:
            subseq = tuple(full[i] for i in sorted(I))
            complement = tuple(full[j] for j in J)
            behavior.setdefault(subseq, {})
            behavior[subseq][complement] = (out, tid)

        # subsequence of the query at I
        query_subseq = tuple(query_tokens[i] for i in sorted(I))

        # only care about pairs where the query is one side
        if query_subseq not in behavior:
            continue

        subseqs = list(behavior.keys())
        for a in subseqs:
            if a == query_subseq:
                continue  # need the *other* side

            # check: one of the two is the query subsequence
            # here a is the other side, query_subseq is one side
            a_q = query_subseq  # the side matching the query
            a_p = a             # the equivalent side

            comps_q = set(behavior[a_q].keys())
            comps_p = set(behavior[a_p].keys())
            shared_complements = comps_q & comps_p
            if not shared_complements:
                continue

            # consistency check
            inconsistent = False
            for c in shared_complements:
                out1, _ = behavior[a_q][c]
                out2, _ = behavior[a_p][c]
                if out1 != out2:
                    inconsistent = True
                    break
            if inconsistent:
                continue

            # collect witness pairs (distinct ids)
            witness_pairs = []
            witness_complements = []
            for c in sorted(shared_complements):
                out1, id1 = behavior[a_q][c]
                out2, id2 = behavior[a_p][c]
                if id1 == id2:
                    continue
                witness_pairs.append([id1, id2])
                witness_complements.append(list(c))

            k = len(witness_complements)
            if k == 0:
                continue

            if verbose:
                print(
                    f"I={sorted(list(I))}, "
                    f"query_subseq={a_q}, equiv={a_p}, "
                    f"k={k}, witnesses={witness_pairs}"
                )

            results.append(
                {
                    "I": sorted(list(I)),
                    "query_subseq": ",".join(map(str, a_q)),
                    "equivalent_subseq": ",".join(map(str, a_p)),
                    "k": k,
                    "shared_complements": witness_complements,
                    "example_pair_ids": witness_pairs,
                }
            )

    return results


# ---------------------------
# small main / test demonstrating the logic
# ---------------------------
def main_demo():
    # Example 1: your scenario where k=2 (two shared complements, outputs match per complement)
    # We'll construct four examples (give them explicit ids)
    # IDs 10..13 to echo your earlier notation
    context1 = [
        {"id": 10, "input": "1,2,7", "output": "L"},   # x1 = (1,2,7) -> L
        {"id": 11, "input": "3,4,7", "output": "L"},   # x1' = (3,4,7) -> L
        {"id": 12, "input": "1,2,8", "output": "B"},   # x2 = (1,2,8) -> B
        {"id": 13, "input": "3,4,8", "output": "B"},   # x2' = (3,4,8) -> B
    ]
    print("=== Demo 1 (expect max_k=2 for I=[0,1] and subseqs '1,2' vs '3,4') ===")
    res1 = find_max_k_equivalence(context1, verbose=True)
    print(res1)
    print()

    # Example 2: if one complement is inconsistent (outputs mismatch), pair is rejected
    context2 = [
        {"id": 0, "input": "1,2,7", "output": "L"},
        {"id": 1, "input": "3,4,7", "output": "X"},  # mismatch here
        {"id": 2, "input": "1,2,8", "output": "B"},
        {"id": 3, "input": "3,4,8", "output": "B"},
    ]
    print("=== Demo 2 (one shared complement mismatches => that subseq pair is discarded) ===")
    res2 = find_max_k_equivalence(context2, verbose=True)
    print(res2)
    print()

    # Example 3: one pair with many distinct witnesses
    context3 = [
        {"id": 0, "input": "1,9,100", "output": 0},
        {"id": 1, "input": "2,9,100", "output": 0},
        {"id": 2, "input": "1,9,101", "output": 0},
        {"id": 3, "input": "2,9,101", "output": 0},
        {"id": 4, "input": "1,9,102", "output": 0},
        {"id": 5, "input": "2,9,102", "output": 0},
    ]
    print("=== Demo 3 (I=[0] subseqs '1' vs '2' have k=3) ===")
    res3 = find_max_k_equivalence(context3, verbose=False)
    print(res3)
    print()


def query_demo():
    # Context: x1=(1,2,7)->L, x1'=(3,4,7)->L, x2=(1,2,8)->B, x2'=(3,4,8)->B
    # (1,2) ≡ (3,4) at I={0,1} with k=2
    context = [
        {"id": 10, "input": "1,2,7", "output": "L"},
        {"id": 11, "input": "3,4,7", "output": "L"},
        {"id": 12, "input": "1,2,8", "output": "B"},
        {"id": 13, "input": "3,4,8", "output": "B"},
    ]
    # Query x3 has (1,2) at I={0,1} — same as one side of the equivalence
    query = {"input": "1,2,9"}
    print("=== Query demo: x3=(1,2,9) — find equivalences where query participates ===")
    results = find_k_equivalences_for_query(context, query, verbose=True)
    for r in results:
        print(r)
    print()


if __name__ == "__main__":
    main_demo()
    query_demo()