from typing import Any, Callable
import torch


class LLMJudgeBuilder:
    """Factory for creating judge(prompt_text)->str callables from local GPU models."""

    @staticmethod
    def from_loaded_model(
        model: Any,
        tokenizer: Any,
        device: str | None = None,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Callable[[str], str]:
        run_device = device
        if run_device is None:
            run_device = "cuda" if torch.cuda.is_available() else "cpu"

        if hasattr(model, "to"):
            model.to(run_device)
        if hasattr(model, "eval"):
            model.eval()

        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        def judge(judge_prompt: str) -> str:
            enc = tokenizer(judge_prompt, return_tensors="pt")
            enc = {k: v.to(run_device) for k, v in enc.items()}

            with torch.no_grad():
                out = model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc.get("attention_mask"),
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    eos_token_id=getattr(tokenizer, "eos_token_id", None),
                    pad_token_id=getattr(tokenizer, "pad_token_id", None),
                )

            prompt_len = enc["input_ids"].shape[-1]
            generated_ids = out[0][prompt_len:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            if text:
                return text
            return tokenizer.decode(out[0], skip_special_tokens=True).strip()

        return judge
