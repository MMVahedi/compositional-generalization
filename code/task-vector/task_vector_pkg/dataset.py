from typing import List
import json

from coverage.demontration_pair import DemoPair
from task_vector_pkg.query import Query


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

    @staticmethod
    def export_to_file(dataset, filepath: str):
        """Export the dataset to a JSON file."""
        data = {
            "coverage_degree": dataset.coverage_degree,
            "queries": [
                {
                    "demonstrations": [
                        {"input": demo.input, "output": demo.output, "id": demo.id, "meta": demo.meta}
                        for demo in query.demonstrations
                    ],
                    "query_demo": {
                        "input": query.query_demo.input,
                        "output": query.query_demo.output,
                        "id": query.query_demo.id,
                        "meta": query.query_demo.meta
                    }
                }
                for query in dataset.queries
            ]
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def from_file(filepath: str):
        """Import a dataset from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        coverage_degree = data["coverage_degree"]
        queries = []
        for q_data in data["queries"]:
            demonstrations = [
                DemoPair(
                    input=d["input"],
                    output=d["output"],
                    id=d["id"],
                    meta=d["meta"]
                )
                for d in q_data["demonstrations"]
            ]
            query_demo = DemoPair(
                input=q_data["query_demo"]["input"],
                output=q_data["query_demo"]["output"],
                id=q_data["query_demo"]["id"],
                meta=q_data["query_demo"]["meta"]
            )
            queries.append(Query(demonstrations, query_demo))
        
        return Dataset(queries, coverage_degree)


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