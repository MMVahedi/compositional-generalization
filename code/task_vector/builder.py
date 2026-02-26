from typing import List
import logging
import torch

from task_vector.task_vector_prompt import TaskVectorPrompt
from task_vector.utils import TaskVectorConfig, TaskVector
from task_vector.extractor import TaskVectorExtractor

class TaskVectorBuilder:
    """Builder for creating task vectors from a dataset of queries."""

    def __init__(self, prompts: List[TaskVectorPrompt], model, cfg: TaskVectorConfig, separator_text: str):
        self.model = model
        self.prompts = prompts
        self.cfg = cfg
        self.separator_text = separator_text

    def build_task_vector(self) -> TaskVector:
        """Extract and average task vectors from a list of queries."""
        logging.info("Starting task vector extraction")
        sep = self.separator_text
        extractor = TaskVectorExtractor(self.model, self.cfg)
        task_vectors = []
        
        for idx, prompt in enumerate(self.prompts):
            task_vec = extractor.extract(prompt, separator_text=sep)
            task_vectors.append(task_vec)
            if (idx + 1) % 20 == 0:
                logging.debug(f"Extracted {idx + 1}/{len(self.prompts)} task vectors")

        if not task_vectors:
            raise RuntimeError("No task vectors extracted; check your queries.")
        
        logging.info(f"Extracted {len(task_vectors)} task vectors, computing average")
        vecs = torch.stack([tv.vector for tv in task_vectors], dim=0)
        avg_vec = vecs.mean(dim=0)
        logging.debug(f"Average vector shape: {avg_vec.shape}")
        
        if self.cfg.normalize == "l2":
            logging.info("Applying L2 normalization to averaged vector")
            avg_vec = avg_vec / (avg_vec.norm(p=2) + 1e-12)

        avg_task_vector = TaskVector(
            vector=avg_vec.detach(),
            layer_idx=task_vectors[0].layer_idx,
            separator_text=sep,
            average_separators=self.cfg.average_separators,
            meta={"n_vectors": len(task_vectors)},
        )
        logging.info(f"Task vector built successfully: shape={avg_vec.shape}, n_vectors={len(task_vectors)}")
        return avg_task_vector
