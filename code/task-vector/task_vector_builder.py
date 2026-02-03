from typing import List
import torch

from icl_task_vectors import TaskVectorExtractor, TaskVector
from query import Query


class TaskVectorBuilder:
    """Builder for creating task vectors from a dataset of queries."""

    def __init__(self, model, cfg, prompt_builder):
        self.model = model
        self.cfg = cfg
        self.prompt_builder = prompt_builder

    def build_task_vector(self, queries: List[Query], sep: str, system_prompt: str) -> TaskVector:
        """Extract and average task vectors from a list of queries."""
        extractor = TaskVectorExtractor(self.model, self.cfg)
        task_vectors = []
        
        for query in queries:
            fewshot_text = query.build_prompt(sep, system_prompt)
            fewshot_enc = self.prompt_builder.encode(fewshot_text, device=self.cfg.device)
            task_vec = extractor.extract(fewshot_enc, separator_text=sep)
            task_vectors.append(task_vec)

        if not task_vectors:
            raise RuntimeError("No task vectors extracted; check your queries.")

        vecs = torch.stack([tv.vector for tv in task_vectors], dim=0)
        avg_vec = vecs.mean(dim=0)
        if self.cfg.normalize == "l2":
            avg_vec = avg_vec / (avg_vec.norm(p=2) + 1e-12)

        avg_task_vector = TaskVector(
            vector=avg_vec.detach(),
            layer_idx=task_vectors[0].layer_idx,
            separator_text=sep,
            average_separators=self.cfg.average_separators,
            meta={"n_vectors": len(task_vectors)},
        )
        return avg_task_vector