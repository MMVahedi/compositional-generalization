from typing import Any, Dict, Optional
import logging
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from task_vector.injector import (
    InjectorBackend,
    Injector,
    TaskVectorConfig,
    GPT2StyleBackend,
    LlamaBackend,
    QwenBackend,
)
from task_vector.task_vector_prompt import TaskVectorPrompt
from dataset.dataset import Dataset
from task_vector.builder import TaskVectorBuilder
from task_vector.utils import Config
from icl.prompt import FewShotPrompt
from evaluation.evaluator import ContainmentEvaluator

class TaskVectorExperiment:
    """Encapsulates all components needed for a task vector experiment."""

    def __init__(self, dataset: Dataset, model_path: str, configs: Config):
        logging.info("Initializing TaskVectorExperiment")
        self.dataset = dataset
        self.configs = configs
        logging.info(f"Dataset size: {len(dataset)} queries")

        # Create TaskVectorConfig from configs
        logging.info("Creating TaskVectorConfig")
        self.task_vector_config = TaskVectorConfig(
            layer_idx=configs.block_idx,
            average_separators=configs.average_separators,
            normalize=configs.normalize,
            alpha=configs.alpha,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        logging.info(f"TaskVectorConfig: layer_idx={self.task_vector_config.layer_idx}, alpha={self.task_vector_config.alpha}, device={self.task_vector_config.device}")
        logging.info(f"TaskVectorConfig: layer_idx={self.task_vector_config.layer_idx}, alpha={self.task_vector_config.alpha}, device={self.task_vector_config.device}")
        
        # Load Model and Tokenizer
        logging.info(f"Loading model and tokenizer from {model_path}")
        self.model, self.tokenizer = self._load_model_and_tokenizer(model_path, str(self.task_vector_config.device))
        logging.info("Model and tokenizer loaded successfully")
        
        # Create TaskVector
        logging.info("Creating task vector prompts from dataset queries")
        self.task_vector_prompts = self._create_prompts()
        logging.info(f"Created {len(self.task_vector_prompts)} task vector prompts")
        
        logging.info("Building task vector from prompts")
        task_vector_builder = TaskVectorBuilder(
            self.task_vector_prompts,
            self.model,
            self.task_vector_config,
            self.configs.sep,
        )
        self.task_vector = task_vector_builder.build_task_vector()
        logging.info(f"Task vector built with shape: {self.task_vector.vector.shape}")
        
        # Create Injector
        logging.info("Selecting injection backend")
        backend = self._choose_backend(self.model)
        logging.info(f"Selected backend: {backend.__class__.__name__}")
        self.injector =  Injector(backend, self.task_vector, self.task_vector_config)
        logging.info("Injector initialized")

        # Create Evaluator 
        logging.info("Initializing evaluator")
        self.evaluator = ContainmentEvaluator()
        logging.info("TaskVectorExperiment initialization complete")

    def _create_prompts(self):
        """Create prompts for all queries in the dataset."""
        logging.debug(f"Creating prompts for {len(self.dataset.queries)} queries")
        prompts = []
        for idx, query in enumerate(self.dataset.queries):
            few_shot_prompt = FewShotPrompt(query, self.configs.sep, self.configs.system_prompt)
            task_vector_prompt = TaskVectorPrompt(few_shot_prompt, self.tokenizer)
            prompts.append(task_vector_prompt)
            if (idx + 1) % 100 == 0:
                logging.debug(f"Created {idx + 1}/{len(self.dataset.queries)} prompts")
        return prompts

    def _load_model_and_tokenizer(self, model_source: str, device: str, local_files_only: bool = True):
        logging.info(f"Loading tokenizer from {model_source}")
        tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=local_files_only)
        logging.info(f"Tokenizer loaded: vocab_size={len(tokenizer)}")
        
        logging.info(f"Loading model from {model_source} to device {device}")
        model = AutoModelForCausalLM.from_pretrained(
            model_source, local_files_only=local_files_only, dtype="float16", low_cpu_mem_usage=True
        ).to(device)
        model.eval()
        logging.info("Model loaded and set to eval mode")

        # Some tokenizers (e.g. GPT-2) have no pad token; set it for generation convenience
        if tokenizer.pad_token_id is None:
            logging.info("Setting pad_token to eos_token")
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def _choose_backend(self, model: nn.Module, prefer: Optional[str] = None) -> InjectorBackend:
        """
        prefer: "llama" | "qwen" | "gpt2" | None
        If None, infer by attribute structure.
        """
        logging.debug("Choosing injection backend")
        if prefer == "gpt2":
            logging.debug("Using GPT2StyleBackend (from preference)")
            return GPT2StyleBackend()
        if prefer == "llama":
            logging.debug("Using LlamaBackend (from preference)")
            return LlamaBackend()
        if prefer == "qwen":
            logging.debug("Using QwenBackend (from preference)")
            return QwenBackend()

        # auto-infer
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            # LLaMA-like (also Qwen2)
            logging.debug("Auto-detected LlamaBackend (model.model.layers)")
            return LlamaBackend()
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            logging.debug("Auto-detected GPT2StyleBackend (model.transformer.h)")
            return GPT2StyleBackend()

        # fallback: QwenBackend tries both
        logging.debug("Using QwenBackend as fallback")
        return QwenBackend()


    def run(self) -> Dict[str, Any]:
        """Run the experiment: extract task vector, evaluate on queries, and report results."""
        logging.info("="*60)
        logging.info("Starting experiment run")
        logging.info("="*60)
        # Build task vector from dataset
        sep = self.configs.sep        
        correct_baseline = 0
        correct_injected = 0
        total = len(self.task_vector_prompts)
        logging.info(f"Total test prompts: {total}")
        
        # Fewshot Injection
        logging.info("Starting task vector injection evaluation")
        skipped = 0
        for idx, prompt in enumerate(self.task_vector_prompts):
            sep_pos = None
            for i, token in enumerate(prompt.input_ids[0]):
                if self.tokenizer.decode(token) == sep:
                    sep_pos = i
                    break
            if sep_pos is None:
                skipped += 1
                continue  # Skip if no separator

            response = self.injector.inject_and_forward(
                model=self.model,
                prompt=prompt,
                inject_position=sep_pos,
                use_cache=False
            )
            if self.evaluator.is_correct(prompt.prompt, response):
                correct_injected += 1
            
            if (idx + 1) % 50 == 0:
                logging.info(f"Injected evaluation progress: {idx + 1}/{total} ({correct_injected}/{idx + 1 - skipped} correct so far)")
        
        if skipped > 0:
            logging.warning(f"Skipped {skipped} prompts due to missing separator")
        logging.info(f"Injected Accuracy: {correct_injected}/{total} = {correct_injected/total:.2%}")

        # Fewshot Baseline
        logging.info("Starting baseline (no injection) evaluation")
        for idx, prompt in enumerate(self.task_vector_prompts):
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
            
            if (idx + 1) % 50 == 0:
                logging.info(f"Baseline evaluation progress: {idx + 1}/{total} ({correct_baseline}/{idx + 1} correct so far)")

        logging.info(f"Baseline Accuracy: {correct_baseline}/{total} = {correct_baseline/total:.2%}")
        logging.info("="*60)
        logging.info("Experiment run completed")
        logging.info("="*60)                        
