"""Module that contains function to visualize context sets"""

import torch
import matplotlib.pyplot as plt

import src.train.training as training

from typing import List, Optional, Tuple

from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from src.dataset.general_dataset import NeuralizerGeneralDataset
from src.dataset.sampler import WeightedTaskSampler
from src.tasks.base_task import BaseTask


def plot_context_set(ctx: torch.Tensor, sup_title: Optional[str] = None) -> None:
    """Utility function to plot a context set

    :param ctx: torch.Tensor -- Context Set
    :param sup_title: Optional[str] -- Figure sup title to pass (Optional)
    :return: None
    """
    n_context = ctx.shape[0]
    fig, axes = plt.subplots(nrows=1, ncols=n_context, figsize=(2 * n_context, 5), sharey=True, sharex=True)
    if sup_title is not None:
        fig.suptitle(sup_title, fontweight="bold", y=0.75, fontsize=16)
    for i, ax in enumerate(axes.flatten()):
        ax.set_title(f"Context Member {i}")
        ax.imshow(to_pil_image(ctx[i, :, :, :]))
        ax.grid(False)
    fig.tight_layout()


def plot_context_set_overlay(ctx_in: torch.Tensor, ctx_out: torch.Tensor, alpha: float = 0.5) -> None:
    """Utility function to plot the context set input and output with an overlay

    :param ctx_in: torch.Tensor -- Input Context
    :param ctx_out: torch.Tensor -- Output Context
    :param alpha: float -- Opacity level
    :return: None
    """
    assert ctx_in.shape[0] == ctx_out.shape[0], "Mismatching number of members in context in and out!"

    n_context = ctx_in.shape[0]
    fig, axes = plt.subplots(nrows=1, ncols=n_context, figsize=(2 * n_context, 5), sharey=True, sharex=True)
    for i, ax in enumerate(axes.flatten()):
        ax.set_title(f"Context Member {i}")
        ax.imshow(to_pil_image(ctx_in[i, :, :, :]))
        ax.imshow(to_pil_image(ctx_out[i, :, :, :]), alpha=alpha)
        ax.grid(False)
    fig.tight_layout()


def plot_all_tasks(
    CFG: dict, tasks: List[BaseTask], context_set_size: int = 8, resize_shape: Tuple[int, int] = (192, 192)
) -> None:
    """Utility function to plot the context set of each task provided in the tasks parameter.
    Will output the context set input and output with a total of context_set_size members.

    :param tasks: List[BaseTask] -- List of tasks to visualize the context sets
    :param context_set_size: int -- Number of members the visualized context sets should have
    :param resize_shape: Tuple[int, int] -- Shape to resize the images to
    :return: None
    """
    for task in tasks:
        dataset = NeuralizerGeneralDataset(
            tasks=[task],
            resize_shape=resize_shape,
            context_set_size=context_set_size,
            augmented_transforms=training.get_augmentation_transformations(CFG),
        )
        sampler = WeightedTaskSampler(dataset)
        dataloader = DataLoader(dataset=dataset, sampler=sampler)
        _, _, ctx_in, ctx_out, _, _ = next(iter(dataloader))

        plot_context_set(ctx_in[0], sup_title=task.get_task_name())
        plot_context_set(ctx_out[0])
