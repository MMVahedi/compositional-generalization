from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from task_vector.task_vector_injector import (
    InjectorBackend,
    Injector,
    TaskVectorConfig,
    GPT2StyleBackend,
    LlamaBackend,
    QwenBackend,
)
from task_vector.task_vector_prompt import TaskVectorPrompt
from dataset.dataset import Dataset
from task_vector.task_vector_builder import TaskVectorBuilder
from task_vector.config import Config
from icl.prompt import FewShotPrompt
from evaluation.evaluator import ContainmentEvaluator

class TaskVectorExperiment:
    """Encapsulates all components needed for a task vector experiment."""

    def __init__(self, dataset: Dataset, model_path: str, configs: Config):
        self.dataset = dataset
        self.configs = configs

        # Create TaskVectorConfig from configs
        self.task_vector_config = TaskVectorConfig(
            layer_idx=configs.block_idx,
            average_separators=configs.average_separators,
            normalize=configs.normalize,
            alpha=configs.alpha,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # Load Model and Tokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer(model_path, str(self.task_vector_config.device))
        
        # Create TaskVector
        self.task_vector_prompts = self._create_prompts()
        task_vector_builder = TaskVectorBuilder(self.task_vector_prompts, self.model, self.task_vector_config)
        self.task_vector = task_vector_builder.build_task_vector()
        
        # Create Injector
        backend = self._choose_backend(self.model)
        self.injector =  Injector(backend, self.task_vector, self.task_vector_config)

        # Create Evaluator 
        self.evaluator = ContainmentEvaluator()

    def _create_prompts(self):
        """Create prompts for all queries in the dataset."""
        prompts = []
        for query in self.dataset.queries:
            few_shot_prompt = FewShotPrompt(query, self.configs.sep, self.configs.system_prompt)
            task_vector_prompt = TaskVectorPrompt(few_shot_prompt, self.tokenizer, self.configs.sep)
            prompts.append(task_vector_prompt)
        return prompts

    def _load_model_and_tokenizer(self, model_source: str, device: str, local_files_only: bool = True):
        tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=local_files_only)
        model = AutoModelForCausalLM.from_pretrained(
            model_source, local_files_only=local_files_only, dtype="float16", low_cpu_mem_usage=True
        ).to(device)
        model.eval()

        # Some tokenizers (e.g. GPT-2) have no pad token; set it for generation convenience
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def _choose_backend(self, model: nn.Module, prefer: Optional[str] = None) -> InjectorBackend:
        """
        prefer: "llama" | "qwen" | "gpt2" | None
        If None, infer by attribute structure.
        """
        if prefer == "gpt2":
            return GPT2StyleBackend()
        if prefer == "llama":
            return LlamaBackend()
        if prefer == "qwen":
            return QwenBackend()

        # auto-infer
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            # LLaMA-like (also Qwen2)
            return LlamaBackend()
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return GPT2StyleBackend()

        # fallback: QwenBackend tries both
        return QwenBackend()


    def run(self) -> Dict[str, Any]:
        """Run the experiment: extract task vector, evaluate on queries, and report results."""
        # Build task vector from dataset
        sep = self.configs.sep        
        correct_baseline = 0
        correct_injected = 0
        total = len(self.task_vector_prompts)
        
        # Fewshot Injection
        for prompt in self.task_vector_prompts:
            sep_pos = None
            for i, token in enumerate(prompt.input_ids[0]):
                if self.tokenizer.decode(token) == sep:
                    sep_pos = i
                    break
            if sep_pos is None:
                continue  # Skip if no separator

            response = self.injector.inject_and_forward(
                model=self.model,
                prompt=prompt,
                inject_position=sep_pos,
                use_cache=False
            )
            if self.evaluator.is_correct(prompt.prompt, response):
                correct_injected += 1

        print(f"Injected Accuracy: {correct_injected}/{total} = {correct_injected/total:.2%}")

        # Fewshot Baseline
        for prompt in self.task_vector_prompts:
            out = self.model.generate(
                input_ids=prompt.input_ids,
                attention_mask=prompt.attention_mask,
                max_new_tokens=self.configs.max_tokens,
                do_sample=False,
                temperature=self.configs.temperature,
                top_k=self.configs.top_k,
                top_p=self.configs.top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            baseline = self.tokenizer.decode(out[0], skip_special_tokens=True)
            
            if self.evaluator.is_correct(prompt.prompt, baseline):
                correct_baseline += 1

        print(f"Baseline Accuracy: {correct_baseline}/{total} = {correct_baseline/total:.2%}")                        
