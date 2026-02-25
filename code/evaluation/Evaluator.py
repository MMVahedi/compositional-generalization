from typing import Callable
import re

from icl.prompt import FewShotPrompt, ZeroShotPrompt

class ContainmentEvaluator:
    """Evaluates correctness by checking if response contains the expected result."""

    def __init__(self, prompt: FewShotPrompt | ZeroShotPrompt, model_response: str):
        self.prompt = prompt
        self.correct_result = self.prompt.get_correct_result()
        self.model_response = model_response

    def is_correct(self) -> bool:
        correct_result = self.correct_result.strip()
        response_text = self.model_response.strip()
        return correct_result.lower() in response_text.lower()


class LLMAssistedEvaluator:
    """Evaluates correctness using an external LLM judge function."""

    def __init__(self, prompt: FewShotPrompt | ZeroShotPrompt, model_response: str, judge: Callable[[str], str]):
        self.prompt = prompt
        self.model_response = model_response
        self.judge = judge

    def _create_judge_prompt(self) -> str:
        return f"Prompt:\n{self.prompt.build()}\n\nModel Response:\n{self.model_response}\n\nIs the model response correct? Answer 'yes' or 'no'."

    def _extract_yes_no(self, judge_output: str) -> bool:
        text = judge_output.strip().lower()

        matches = re.findall(r"\b(yes|no)\b", text)
        if matches:
            return matches[-1] == "yes"

        if "correct" in text and "incorrect" not in text:
            return True
        if "incorrect" in text or "wrong" in text:
            return False

        raise ValueError(f"Could not extract yes/no from judge output: {judge_output}")

    def is_correct(self) -> bool:
        judge_output = self.judge(self._create_judge_prompt())
        return self._extract_yes_no(judge_output)

