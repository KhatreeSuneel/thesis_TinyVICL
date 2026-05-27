"""
Defines and loads evaluation objects for the unseen task evaluation.
Six models are evaluated: 1M and 5M architecture variants, each trained
with one of three loss functions: MSE, MSE+SSIM, and task_sp.
Each architecture variant uses its own config (CFG_1M / CFG_5M).
"""

import logging
import torch

from typing import List, Union

from scripts.eval_base import EvalObject

logger = logging.getLogger(__name__)


def get_eval_objects(
    CFG_1M: dict,
    CFG_5M: dict,
    device: Union[str, torch.device],
) -> List[EvalObject]:
    models = [
        # -----------------------------
        # RANDOM BASELINE
        # -----------------------------
        EvalObject(
            name="Random",
            checkpoint_path=None,
            device=device,
            CFG=CFG_1M,
        ),
        # -----------------------------
        # 1M PARAM MODELS
        # -----------------------------
        EvalObject(
            name="Neuralizer_General_1M_MSE",
            checkpoint_path="path/to/1m/mse.ckpt",
            device=device,
            CFG=CFG_1M,
        ),
        EvalObject(
            name="Neuralizer_General_1M_MSE+SSIM",
            checkpoint_path="path/to/1m/mse_ssim.ckpt",
            device=device,
            CFG=CFG_1M,
        ),
        EvalObject(
            name="Neuralizer_General_1M_task_sp",
            checkpoint_path="path/to/1m/task.ckpt",
            device=device,
            CFG=CFG_1M,
        ),
        # -----------------------------
        # 5M PARAM MODELS
        # -----------------------------
        EvalObject(
            name="Neuralizer_General_5M_MSE",
            checkpoint_path="path/to/5m/mse.ckpt",
            device=device,
            CFG=CFG_5M,
        ),
        EvalObject(
            name="Neuralizer_General_5M_MSE+SSIM",
            checkpoint_path="path/to/5m/mse_ssim.ckpt",
            device=device,
            CFG=CFG_5M,
        ),
        EvalObject(
            name="Neuralizer_General_5M_task_sp",
            checkpoint_path="path/to/5m/task.ckpt",
            device=device,
            CFG=CFG_5M,
        ),
    ]
    return models