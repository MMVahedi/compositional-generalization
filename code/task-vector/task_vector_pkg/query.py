from typing import List

from coverage.demonstration_pair import DemoPair


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
        query_covered_demos = self.query_demo.covered_demos
        counter = 0
        for demo in query_covered_demos:
            if demo in self.demonstrations:
                counter += 1
        return counter

    def build_prompt(self, sep: str, system_prompt: str | None = None) -> str:
        """Build the few-shot prompt from demonstrations and query."""
        chunks = [f"{p.input}{sep}{p.output}" for p in self.demonstrations]
        chunks.append(f"{self.query_demo.input}{sep}")
        user_prompt = ",".join(chunks)
        return user_prompt if system_prompt is None else f"{system_prompt}\n{user_prompt}"

    def __repr__(self):
        return f"Query(query_input='{self.query_demo.input}', num_demos={len(self.demonstrations)})"
