from typing import Any, Dict
import torch
from icl_task_vectors import choose_backend, Injector, TaskVectorConfig
from task_vector_pkg.dataset import Dataset
from task_vector_pkg.task_vector import TaskVectorBuilder
from task_vector_pkg.config import Config


class Experiment:
    """Encapsulates all components needed for a task vector experiment."""

    def __init__(self, dataset: Dataset, model: Any, configs: Config, prompt_builder: Any):
        self.dataset = dataset
        self.model = model
        self.configs = configs
        self.prompt_builder = prompt_builder
        
        # Create TaskVectorConfig from configs
        self.cfg = TaskVectorConfig(
            layer_idx=configs.block_idx,
            average_separators=configs.average_separators,
            normalize=configs.normalize,
            alpha=configs.alpha,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # Create TaskVectorBuilder
        self.task_vector_builder = TaskVectorBuilder(model, self.cfg, prompt_builder)

    def run(self) -> Dict[str, Any]:
        """Run the experiment: extract task vector, evaluate on queries, and report results."""
        # Build task vector from dataset
        sep = self.configs.sep
        system_prompt = self.configs.system_prompt
        task_vector = self.task_vector_builder.build_task_vector(self.dataset.queries, sep, system_prompt)
        
        # Setup for inference
        tokenizer = self.task_vector_builder.prompt_builder.tokenizer
        backend = choose_backend(self.model)
        injector = Injector(backend, self.task_vector_builder.cfg)
        
        results = []
        correct_baseline = 0
        correct_injected = 0
        total = len(self.dataset.queries)
        
        for query in self.dataset.queries:
            # Build query prompt
            query_text = query.build_prompt(sep, system_prompt)
            query_enc = self.task_vector_builder.prompt_builder.encode(query_text, device=self.task_vector_builder.cfg.device)
            
            # Find separator position
            sep_pos = None
            for i, token in enumerate(query_enc.input_ids[0]):
                if tokenizer.decode(token) == sep:
                    sep_pos = i
                    break
            if sep_pos is None:
                continue  # Skip if no separator
            
            # Baseline generation (without injection)
            baseline_output = self._generate_text(query_enc, tokenizer)
            
            # Injected generation
            injector.inject_and_forward(
                model=self.model, 
                prompt=query_enc, 
                task_vector=task_vector, 
                inject_position=sep_pos, 
                use_cache=False
            )
            injected_output = self._generate_text(query_enc, tokenizer)
            
            # Validate results
            expected = query.query_demo.output
            baseline_correct = self.evaluate_output(baseline_output, expected)
            injected_correct = self.evaluate_output(injected_output, expected)
            
            if baseline_correct:
                correct_baseline += 1
            if injected_correct:
                correct_injected += 1
            
            results.append({
                'query_input': query.query_demo.input,
                'expected_output': expected,
                'baseline_output': baseline_output,
                'injected_output': injected_output,
                'baseline_correct': baseline_correct,
                'injected_correct': injected_correct
            })
        
        # Report results
        report = {
            'total_queries': total,
            'baseline_accuracy': correct_baseline / total if total > 0 else 0,
            'injected_accuracy': correct_injected / total if total > 0 else 0,
            'improvement': (correct_injected - correct_baseline) / total if total > 0 else 0,
            'results': results
        }
        
        print("Experiment Results:")
        print(f"Total Queries: {report['total_queries']}")
        print(f"Baseline Accuracy: {report['baseline_accuracy']:.2%}")
        print(f"Injected Accuracy: {report['injected_accuracy']:.2%}")
        print(f"Improvement: {report['improvement']:.2%}")
        
        return report

    def _generate_text(self, enc, tokenizer):
        """Generate text from encoded input."""
        max_tokens = self.configs.max_tokens
        temperature = self.configs.temperature
        top_k = self.configs.top_k
        top_p = self.configs.top_p
        
        out = self.model.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(out[0], skip_special_tokens=True)

    def evaluate_output(self, model_output: str, ground_truth: str) -> bool:
        """Evaluate if the model output matches the ground truth."""
        return model_output.strip() == ground_truth.strip()

    def __repr__(self):
        return f"Experiment(dataset={self.dataset}, model={type(self.model).__name__}, configs={self.configs})"
