from typing import Callable
import re

from icl.prompt import FewShotPrompt, ZeroShotPrompt

class ContainmentEvaluator:
    """Evaluates correctness by checking if response contains the expected result."""

    @staticmethod
    def _text_after_kth_separator(text: str, separator: str, k: int) -> str | None:
        if k <= 0:
            return None
        pos = -1
        start = 0
        for _ in range(k):
            pos = text.find(separator, start)
            if pos == -1:
                return None
            start = pos + len(separator)
        return text[start:]

    @staticmethod
    def _extract_first_number(text: str) -> str | None:
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        return match.group(0) if match else None

    def is_correct(self, prompt: FewShotPrompt | ZeroShotPrompt, model_response: str) -> bool:
        correct_result = prompt.get_correct_result().strip()

        shots = prompt.query.number_of_shots if isinstance(prompt, FewShotPrompt) else 0
        kth_separator = shots + 1

        answer_region = self._text_after_kth_separator(model_response, prompt.separator, kth_separator)
        if answer_region is None:
            return False

        pred_num = self._extract_first_number(answer_region)
        gold_num = self._extract_first_number(correct_result)

        if pred_num is not None and gold_num is not None:
            return pred_num == gold_num

        pred_token = answer_region.strip().split(",")[0].split()[0] if answer_region.strip() else ""
        gold_token = correct_result.split(",")[0].split()[0] if correct_result else ""
        return pred_token.lower() == gold_token.lower()


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

