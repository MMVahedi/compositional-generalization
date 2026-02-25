from dataset.query import Query
from dataset.demonstration_pair import DemoPair


class FewShotPrompt:
    """Builds a few-shot prompt from a Query (demonstrations + question)."""

    def __init__(self, query: Query, separator: str, system_prompt: str | None = None):
        self.query = query
        self.separator = separator
        self.system_prompt = system_prompt

    def _attach_system_prompt(self, user_prompt: str) -> str:
        return user_prompt if self.system_prompt is None else f"{self.system_prompt}\n{user_prompt}"

    def build(self) -> str:
        chunks = [f"{p.input}{self.separator}{p.output}" for p in self.query.demonstrations]
        chunks.append(f"{self.query.query_demo.input}{self.separator}")
        user_prompt = ",".join(chunks)
        return self._attach_system_prompt(user_prompt)

    def get_correct_result(self) -> str:
        return self.query.get_result()


class ZeroShotPrompt:
    """Builds a zero-shot prompt from a single DemoPair question item."""

    def __init__(self, query_demo: DemoPair, separator: str, system_prompt: str | None = None):
        self.query_demo = query_demo
        self.separator = separator
        self.system_prompt = system_prompt

    def _attach_system_prompt(self, user_prompt: str) -> str:
        return user_prompt if self.system_prompt is None else f"{self.system_prompt}\n{user_prompt}"

    def build(self) -> str:
        user_prompt = f"{self.query_demo.input}{self.separator}"
        return self._attach_system_prompt(user_prompt)

    def get_correct_result(self) -> str:
        return self.query_demo.output
