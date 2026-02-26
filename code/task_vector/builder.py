from typing import List
import torch

from task_vector.task_vector_prompt import TaskVectorPrompt
from task_vector.utils import Config, TaskVector
from task_vector.extractor import TaskVectorExtractor

class TaskVectorBuilder:
    """Builder for creating task vectors from a dataset of queries."""

    def __init__(self, prompts: List[TaskVectorPrompt], model, cfg: Config):
        self.model = model
        self.prompts = prompts
        self.cfg = cfg

    def build_task_vector(self) -> TaskVector:
        """Extract and average task vectors from a list of queries."""
        sep = self.cfg.sep
        extractor = TaskVectorExtractor(self.model, self.cfg)
        task_vectors = []
        
        for prompt in self.prompts:
            task_vec = extractor.extract(prompt, separator_text=sep)
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
