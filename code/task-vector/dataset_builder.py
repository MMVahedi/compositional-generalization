from typing import List
import itertools

from coverage.demontration_pair import DemoPair
from query import Query


class DatasetBuilder:
    """Builder for creating all possible Query objects from a list of demo pairs."""

    def build_queries(self, demos: List[DemoPair], num_shots: int) -> List[Query]:
        """Generate all possible Query objects.

        For each possible query_demo, select combinations of num_shots demonstrations
        from the remaining demos, and for each combination, generate all permutations
        to account for different orders.
        """
        queries = []
        for query_index in range(len(demos)):
            query_demo = demos[query_index]
            remaining = [d for i, d in enumerate(demos) if i != query_index]
            
            if len(remaining) < num_shots:
                continue  # Not enough demos for demonstrations
            
            # Generate all combinations of num_shots from remaining
            for combo in itertools.combinations(remaining, num_shots):
                # Generate all permutations of each combination
                for perm in itertools.permutations(combo):
                    queries.append(Query(list(perm), query_demo))
        
        return queries