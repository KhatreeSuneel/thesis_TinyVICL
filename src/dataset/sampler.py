"""
Module containing the custom PyTorch Sampler
"""

import torch
import logging
import numpy as np

from torch.utils.data import Sampler

from src.dataset.general_dataset import NeuralizerGeneralDataset

logger = logging.getLogger(__name__)


class WeightedTaskSampler(Sampler):
    """
    Custom weighted sampler to sample a random task with a random valid index sample.
    The weighting ensures that minority tasks are getting oversampled, such that each task (independent of the
    sample size) is forwarded equally times.

    Output is, e.g. of shape (0, 499), whereas 0 responds to task index 0 and 499 to sample index of metadata_df.
    """

    def __init__(self, dataset: NeuralizerGeneralDataset, replacement: bool = True, test_mode: bool = False):
        super().__init__()
        self.dataset = dataset
        self.replacement = replacement
        self.test_mode = test_mode

        self.task_indices = []
        self.weights = []

        # Create task_idx and sample_idx pairs; Also extract weights based on # of samples per task dataset
        for task_idx, task in enumerate(dataset.tasks):
            task_length = len(task.get_data())
            task_weight = 1 / task_length
            self.weights.extend([task_weight for _ in range(task_length)])
            task_indices = list(task.get_data().index)
            self.task_indices.extend([(task_idx, idx) for idx in task_indices])

        # Normalize weights and parse to torch tensor
        self.weights = np.array(self.weights)
        self.weights = self.weights / self.weights.sum()
        self.weights = torch.from_numpy(self.weights)

        self.num_samples = len(self.task_indices)

    def __iter__(self):
        # In test mode, return all indices in order; Otherwise return random indices weighted by the task length
        if self.test_mode:
            if len(self.dataset.tasks) == 1:
                task = self.dataset.tasks[0]
                task_indices = list(task.get_data().index)
                task_idx = 0
                return iter([(task_idx, idx) for idx in task_indices])
            else:
                raise ValueError("Test mode expects a single task.")
        else:
            # Sample weighted indices and return the iter object
            indices = torch.multinomial(input=self.weights, num_samples=self.num_samples, replacement=self.replacement)
            return iter([self.task_indices[i] for i in indices])

    def __len__(self):
        # In test mode, the length is simply the number of samples in the single test task
        if self.test_mode and len(self.dataset.tasks) == 1:
            return len(self.dataset.tasks[0].get_data())
        return self.num_samples
