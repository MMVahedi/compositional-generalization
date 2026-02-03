from typing import List
import itertools

from coverage.demontration_pair import DemoPair
from query import Query


class Dataset:
    """A dataset containing queries with a specific coverage degree."""

    def __init__(self, queries: List[Query], coverage_degree: int):
        self.queries = queries
        self.coverage_degree = coverage_degree

    def __len__(self):
        return len(self.queries)

    def __iter__(self):
        return iter(self.queries)

    def __repr__(self):
        return f"Dataset(coverage_degree={self.coverage_degree}, num_queries={len(self.queries)})"


class DatasetBuilder:
    """Builder for creating all possible Query objects from a list of demo pairs."""

    def __init__(self, demos: List[DemoPair], num_shots: int):
        self.demos = demos
        self.num_shots = num_shots
        self.queries = self._build_all_queries()

    def _build_all_queries(self) -> List[Query]:
        """Generate all possible Query objects.

        For each possible query_demo, select combinations of num_shots demonstrations
        from the remaining demos, and for each combination, generate all permutations
        to account for different orders.
        """
        queries = []
        for query_index in range(len(self.demos)):
            query_demo = self.demos[query_index]
            remaining = [d for i, d in enumerate(self.demos) if i != query_index]
            
            if len(remaining) < self.num_shots:
                continue  # Not enough demos for demonstrations
            
            # Generate all combinations of num_shots from remaining
            for combo in itertools.combinations(remaining, self.num_shots):
                # Generate all permutations of each combination
                for perm in itertools.permutations(combo):
                    queries.append(Query(list(perm), query_demo))
        
        return queries

    def get_dataset(self, coverage_degree: int) -> Dataset:
        """Return a Dataset object containing all queries that have the specified coverage degree."""
        filtered_queries = [q for q in self.queries if q.coverage_degree == coverage_degree]
        return Dataset(filtered_queries, coverage_degree)