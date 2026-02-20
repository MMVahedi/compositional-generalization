from typing import Sequence

from coverage.demonstration_pair import DemoPair


def build_fewshot_prompt(pairs: Sequence[DemoPair], query_x: str, sep: str, system_prompt: str | None = None) -> str:
    """Build a compact few-shot prompt where each demo is a single line:
    <input><sep><output>
    and the query is <query_x><sep>
    """
    chunks = [f"{p.input}{sep}{p.output}" for p in pairs]
    chunks.append(f"{query_x}{sep}")
    user_prompt = ",".join(chunks)
    return user_prompt if system_prompt is None else f"{system_prompt}\n{user_prompt}"


def build_query_prompt(query_x: str, sep: str, system_prompt: str | None = None) -> str:
    user_prompt = f"{query_x}{sep}"
    return user_prompt if system_prompt is None else f"{system_prompt}\n{user_prompt}"