from __future__ import annotations

from typing import List, Dict

from ai_bot import MultiLLM, ModelSpec

# ------------------------------------------------------------------
# System prompt A: explicit rule-inference framing
# ------------------------------------------------------------------
SYSTEM_PROMPT_RULE = """\
You are solving a sequence-to-label mapping task.
You will be given a set of input-output examples that demonstrate a pattern, \
followed by a query input whose label you must predict.

Rules:
1. Study the examples carefully to infer the mapping rule.
2. Apply the rule to the query input.
3. Respond with ONLY the output label — no explanation, no reasoning, no punctuation, \
nothing else. A single token answer is expected.\
"""

# ------------------------------------------------------------------
# System prompt B: neutral framing — no mention of rules or patterns
# ------------------------------------------------------------------
SYSTEM_PROMPT_NEUTRAL = """\
You will be shown a list of examples. Each example has an input and an output.
After the examples you will see a query input.

Respond with ONLY the output for the query — no explanation, no extra text, \
just the output value.\
"""


def build_user_prompt(context: List[Dict], query: Dict) -> str:
    """
    Format the ICL prompt from a context (list of labelled examples) and a query.

    Each context item must have at least "input" and "output" keys.
    The query must have at least an "input" key.

    Example context item : {"id": 10, "input": "1,2,7", "output": "L"}
    Example query        : {"input": "1,2,9"}
    """
    lines: List[str] = ["Examples:"]
    for example in context:
        lines.append(f"  Input: {example['input']}  →  Output: {example['output']}")

    lines.append("")
    lines.append("Query:")
    lines.append(f"  Input: {query['input']}  →  Output:")

    return "\n".join(lines)


def icl_query(
    llm: MultiLLM,
    spec: ModelSpec,
    context: List[Dict],
    query: Dict,
    system_prompt: str = SYSTEM_PROMPT_RULE,
    max_tokens: int = 16,
) -> str:
    """
    Run a single ICL query using the given model.

    Parameters
    ----------
    llm           : MultiLLM instance (already initialised with an API key)
    spec          : ModelSpec selecting which model to use
    context       : labelled demonstration examples, e.g.
                    [{"id": 10, "input": "1,2,7", "output": "L"}, ...]
    query         : unlabelled query, e.g. {"input": "1,2,9"}
    system_prompt : which system prompt to use; defaults to SYSTEM_PROMPT_RULE.
                    Pass SYSTEM_PROMPT_NEUTRAL for the neutral framing.
    max_tokens    : cap on the model's response length (default 16 is enough for a label)

    Returns
    -------
    The model's raw response string (should be just the predicted label).
    """
    user_text = build_user_prompt(context, query)
    response = llm.chat(
        spec=spec,
        user_text=user_text,
        system_text=system_prompt,
        max_tokens=max_tokens,
    )
    return response.strip()


# ---------------------------
# quick smoke-test / demo
# ---------------------------
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    METIS_API_KEY = os.environ["METIS_API_KEY"]
    llm = MultiLLM(METIS_API_KEY)
    models = llm.register_models()

    context = [
        {"id": 10, "input": "1,2,7", "output": "L"},
        {"id": 11, "input": "3,4,7", "output": "L"},
        {"id": 12, "input": "1,2,8", "output": "B"},
        {"id": 13, "input": "3,4,8", "output": "B"},
    ]
    query = {"input": "1,2,9"}

    print("=== User prompt ===")
    print(build_user_prompt(context, query))
    print()

    for label, prompt in [
        ("rule framing   ", SYSTEM_PROMPT_RULE),
        ("neutral framing", SYSTEM_PROMPT_NEUTRAL),
    ]:
        answer = icl_query(llm, models["gpt4o_mini"], context, query, system_prompt=prompt)
        print(f"[{label}]  Model answer: {answer!r}")
