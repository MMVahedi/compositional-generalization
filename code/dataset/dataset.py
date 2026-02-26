from typing import List
import logging
import json
from dataset.query import Query
from dataset.demonstration_pair import DemoPair
from function.function import Function

class Dataset:
    """A dataset containing queries with a specific coverage degree."""

    def __init__(self, queries: List[Query], coverage_degree: int):
        self.queries: List[Query] = queries
        self.coverage_degree = coverage_degree

    def __len__(self):
        return len(self.queries)

    def __iter__(self):
        return iter(self.queries)

    def __repr__(self):
        return f"Dataset(coverage_degree={self.coverage_degree}, num_queries={len(self.queries)})"

    @staticmethod
    def export_to_file(dataset: 'Dataset', filepath: str):
        """Export the dataset to a JSON file."""
        data = {
            "coverage_degree": dataset.coverage_degree,
            "queries": [
                {
                    "demonstrations": [DemoPair.to_dict(demo) for demo in query.demonstrations],
                    "query_demo": DemoPair.to_dict(query.query_demo)
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
            demonstrations = [DemoPair.from_dict(d) for d in q_data["demonstrations"]]
            query_demo = DemoPair.from_dict(q_data["query_demo"])
            queries.append(Query(demonstrations, query_demo))
        
        return Dataset(queries, coverage_degree)


class DatasetBuilder:
    """Builder for creating all possible Query objects from a list of demo pairs."""

    def __init__(self, function: Function, num_shots: int, allow_reuse: bool = True):
        self.demos = function.demo_pairs
        self.num_shots = num_shots
        self.allow_reuse = allow_reuse
        self.queries: List[Query] = []
        self.last_generated_count: int = 0
        self.last_filtered_count: int = 0

    def _iter_queries_with_reuse(self):
        """Yield all possible Query objects (original behavior, demos can be reused)."""
        import itertools

        for query_index in range(len(self.demos)):
            query_demo = self.demos[query_index]
            remaining = [d for i, d in enumerate(self.demos) if i != query_index]

            if len(remaining) < self.num_shots:
                continue

            for combo in itertools.combinations(remaining, self.num_shots):
                yield Query(list(combo), query_demo)

    def _iter_queries_without_reuse(self):
        """Yield Query objects without reusing demonstrations across queries."""
        if self.num_shots <= 0:
            raise ValueError("num_shots must be greater than 0")

        block_size = self.num_shots + 1
        full_blocks = len(self.demos) // block_size

        for block_idx in range(full_blocks):
            start = block_idx * block_size
            block = self.demos[start:start + block_size]
            demonstrations = block[:self.num_shots]
            query_demo = block[self.num_shots]
            yield Query(demonstrations, query_demo)

        unused = len(self.demos) - full_blocks * block_size
        if unused > 0:
            logging.info(f"Skipped {unused} trailing demos that could not form a full query block")

    def _build_all_queries_with_reuse(self) -> List[Query]:
        """Generate all possible Query objects (original behavior, demos can be reused)."""
        logging.info(f"Building queries WITH reuse, num_shots={self.num_shots}, demos={len(self.demos)}")
        queries = list(self._iter_queries_with_reuse())
        logging.info(f"Built {len(queries)} total queries (with reuse)")
        return queries

    def _build_all_queries_without_reuse(self) -> List[Query]:
        """Generate Query objects without reusing demonstrations across queries.

        Each query consumes a disjoint block of (num_shots + 1) demos:
        - first num_shots demos are used as demonstrations
        - last demo is used as query_demo
        """
        logging.info(f"Building queries with num_shots={self.num_shots} from {len(self.demos)} demos")
        queries = list(self._iter_queries_without_reuse())
        
        logging.info(f"Built {len(queries)} total queries (without reuse)")
        return queries

    def get_dataset(self, coverage_degree: int) -> Dataset:
        """Build and return only queries that match coverage_degree.

        Queries are generated lazily and filtered immediately to reduce peak memory usage.
        """
        logging.info(f"Building and filtering queries for coverage_degree={coverage_degree}")

        query_iter = self._iter_queries_with_reuse() if self.allow_reuse else self._iter_queries_without_reuse()

        filtered_queries: List[Query] = []
        generated_count = 0
        for query in query_iter:
            generated_count += 1
            if generated_count % 1000 == 0:
                logging.info(f"Generated queries so far: {generated_count}")
            if query.coverage_degree == coverage_degree:
                filtered_queries.append(query)

        self.last_generated_count = generated_count
        self.last_filtered_count = len(filtered_queries)
        self.queries = filtered_queries

        logging.info(
            f"Found {self.last_filtered_count}/{self.last_generated_count} queries with coverage_degree={coverage_degree}"
        )
        return Dataset(filtered_queries, coverage_degree)
