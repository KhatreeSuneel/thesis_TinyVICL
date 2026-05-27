""" Module handling the custom General Dataset used for Neuralizer inheriting from torch.utils.data.Dataset"""


import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from PIL.Image import Resampling
from torchvision.transforms import v2
from torchvision.transforms.functional import pil_to_tensor
from src.utils.colors_utils import colorize_with_default_label
from src.tasks.base_task import BaseTask
from src.utils.label_aware_context import ContextPrecomputeManager
from typing import Tuple, Optional, List

class NeuralizerGeneralDataset(Dataset):
    def __init__(self, tasks, resize_shape=(256,256), context_set_size=7,
                 augmented_transforms=None, test_mode=False, cover_query_labels=True):
        self.tasks = tasks
        self.resize_shape = resize_shape
        self.context_set_size = context_set_size
        self.augmented_transforms = augmented_transforms
        self.test_mode = test_mode
        self.cover_query_labels = cover_query_labels

        # Manager for CSV precompute
        self.manager = ContextPrecomputeManager(save_dir="configs/precomputed_contexts",
                                        resize_shape=resize_shape,
                                        context_set_size=context_set_size)

        # Load or compute CSVs for segmentation tasks
        self.task_csvs = {}
        for i, task in enumerate(tasks):
            if task.get_task_name().lower() == "segmentation" and cover_query_labels:
                csv_path = self.manager.ensure_precomputed(task, i)
                self.task_csvs[i] = self.manager.load_csv(csv_path)

    def __len__(self):
        return sum(len(task.get_data()) for task in self.tasks)

    def __getitem__(self, idx):
        task_idx, sample_idx = idx
        task = self.tasks[task_idx]
        df = task.get_data()
        lossfun = task.get_lossfun()
        is_segmentation = task.get_task_name().lower() == "segmentation"

        # Query image and label
        image_path = os.path.join(df.loc[sample_idx, 'image_dir'], f"{df.loc[sample_idx, 'image_name']}{df.loc[sample_idx, 'image_suffix']}")
        label_path = os.path.join(df.loc[sample_idx, 'label_dir'], f"{df.loc[sample_idx, 'label_name']}{df.loc[sample_idx, 'label_suffix']}")
        image = Image.open(image_path).resize(self.resize_shape, Resampling.BICUBIC)
        label = Image.open(label_path)
        if is_segmentation:
            label = colorize_with_default_label(label).resize(self.resize_shape, Resampling.NEAREST)
        else:
            label = label.resize(self.resize_shape, Resampling.NEAREST)
        
        # Context
        context_in, context_out = self._load_context(df, sample_idx, task)
        
        image = _normalize_image(image)
        label = _normalize_image(label)

        # Augment
        if self.augmented_transforms and ("rotation" not in task.get_task_name().lower()):
            concat = torch.cat([image, label], dim=0)
            transformed = self.augmented_transforms(concat)
            image = transformed[:image.shape[0]]
            label = transformed[label.shape[0]:]

        return image, label, context_in, context_out, lossfun, task.__class__.__name__

    def _load_context(self, df, sample_idx, task) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load a context set for a given sample.
        - For segmentation tasks: use segmentation-specific logic (colorize labels, optional CSV logic)
        - For other tasks: keep context set sampling as before
        """
        task_name = task.get_task_name().lower()
        valid_indices = df.index.tolist()
        valid_indices.remove(sample_idx)
        selected_indices = set()

        # -------------------------------
        # SEGMENTATION TASK With CSV
        # -------------------------------
        if task_name == "segmentation" and self.cover_query_labels:
            if hasattr(self, "task_csvs") and task in self.task_csvs:
                row = self.task_csvs[task].loc[self.task_csvs[task]["query_idx"] == sample_idx]
                if not row.empty:
                    indices_str = row["context_indices"].values[0]
                    if isinstance(indices_str, str) and indices_str.strip("[]"):
                        indices = [int(x.strip()) for x in indices_str.strip("[]").split(",")]
                        selected_indices.update(indices)

            # Fill remaining context indices randomly
            remaining = self.context_set_size - len(selected_indices)
            if remaining > 0:
                remaining_candidates = list(set(valid_indices) - selected_indices)
                if remaining_candidates:
                    extra = np.random.choice(
                        remaining_candidates,
                        size=min(remaining, len(remaining_candidates)),
                        replace=False
                    )
                    selected_indices.update(extra)

            selected_indices = list(selected_indices)[:self.context_set_size]

        # -------------------------------
        # All other tasks (includes segmentation if coverquerylabels set to false)
        # -------------------------------
        else:
            if self.test_mode is None:
                selected_indices = np.random.choice(valid_indices, size=self.context_set_size, replace=False)
            else:
                rng = np.random.default_rng(seed=42)
                selected_indices = rng.choice(valid_indices, size=self.context_set_size, replace=False)
            selected_indices = list(selected_indices)

        # -------------------------------
        # LOAD CONTEXT IMAGES AND LABELS
        # -------------------------------
        context_in, context_out = [], []
        for idx in selected_indices:
            ctx_img_path = os.path.join(
                df.loc[idx, 'image_dir'],
                f"{df.loc[idx, 'image_name']}{df.loc[idx, 'image_suffix']}"
            )
            ctx_label_path = os.path.join(
                df.loc[idx, 'label_dir'],
                f"{df.loc[idx, 'label_name']}{df.loc[idx, 'label_suffix']}"
            )

            ctx_img = Image.open(ctx_img_path).resize(size=self.resize_shape, resample=Resampling.BICUBIC)

            if task_name == "segmentation":
                ctx_label = Image.open(ctx_label_path)
                ctx_label = colorize_with_default_label(ctx_label).resize(size=self.resize_shape, resample=Resampling.NEAREST)
            else:
                ctx_label = Image.open(ctx_label_path).resize(size=self.resize_shape, resample=Resampling.NEAREST)

            ctx_img = pil_to_tensor(ctx_img)
            ctx_label = pil_to_tensor(ctx_label)

            # Optional augmentations
            if (self.augmented_transforms is not None) and ("rotation" not in task_name):
                combined = torch.cat([ctx_img, ctx_label], dim=0)
                transformed = self.augmented_transforms(combined)
                ctx_img = transformed[: ctx_img.shape[0], ...]
                ctx_label = transformed[ctx_label.shape[0] :, ...]

            context_in.append(_normalize_image(ctx_img))
            context_out.append(_normalize_image(ctx_label))

        context_in = torch.stack(context_in, dim=0)
        context_out = torch.stack(context_out, dim=0)

        return context_in, context_out



def _normalize_image(x):
    base_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=False),
        v2.Normalize([0,0,0],[255,255,255])
    ])
    return base_transforms(x)