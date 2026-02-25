from typing import List
from dataset.demonstration_pair import DemoPair


class Query:
    """Represents a query with associated demonstrations and coverage degree."""
    def __init__(self, demonstrations: List[DemoPair], query_demo: DemoPair):
        self.demonstrations = demonstrations
        self.query_demo = query_demo

    @property
    def number_of_shots(self) -> int:
        return len(self.demonstrations)
    
    @property
    def coverage_degree(self) -> int:
        from coverage.simple.determine_coverage import max_coverage_degree_pair_substitution_wrapper
        return max_coverage_degree_pair_substitution_wrapper(self)

    def get_result(self) -> str:
        return self.query_demo.output

    def __repr__(self):
        return f"Query(query_input='{self.query_demo.input}', num_demos={len(self.demonstrations)})"
