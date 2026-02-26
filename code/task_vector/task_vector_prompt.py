from typing import List, Union
from icl.prompt import FewShotPrompt

import torch


class TaskVectorPrompt:
    """
    Locates separator occurrences robustly even when separator_text is multi-token.
    We define the separator position as the *last token* of the separator subsequence.
    """

    def __init__(self, prompt: FewShotPrompt, tokenizer):
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.sep_token_ids = tokenizer.encode(prompt.separator, add_special_tokens=False)
        if len(self.sep_token_ids) == 0:
            raise ValueError("separator_text produced 0 tokens; choose a different separator.")
        self.input_ids, self.attention_mask, self.separator_positions = self._encode(self.prompt.build())

    @staticmethod
    def find_subsequence_starts(sequence: List[int], subseq: List[int]) -> List[int]:
        if len(subseq) == 0 or len(sequence) < len(subseq):
            return []
        hits = []
        m = len(subseq)
        for i in range(0, len(sequence) - m + 1):
            if sequence[i:i + m] == subseq:
                hits.append(i)
        return hits

    def _encode(self, prompt: str, device: Union[str, torch.device] = "cpu"):
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        seq = input_ids[0].tolist()
        starts = TaskVectorPrompt.find_subsequence_starts(seq, self.sep_token_ids)
        # Use last token index of the separator subsequence
        positions = [s + (len(self.sep_token_ids) - 1) for s in starts]
        return input_ids, attention_mask, positions
