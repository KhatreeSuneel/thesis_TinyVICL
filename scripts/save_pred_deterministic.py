"""
Standalone script to load a model and save multiple deterministic prediction results
from the test dataloader to assets/predictions.

Run via:
python3 scripts/new_save_pred.py --checkpoint path_to_ckpt --config path_to_config
"""

import re
import torch
import logging
import random
import pandas as pd
import numpy as np
import os

import src.utils.utils as utils
import src.data.dataloader as dataloader
import src.tasks.taskloader as taskloader
import src.train.training as training

from src.tasks.base_task import BaseTask
from src.neuralizer.lightning_model import LightningModel
from src.visualizations.prediction_visualizer import plot_prediction_batch
from src.utils.utils import get_date_and_time_as_string
from src.dataset.general_dataset import NeuralizerGeneralDataset
from src.dataset.sampler import WeightedTaskSampler

from tqdm import tqdm
from typing import List
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# 🔹 Deterministic Seed Setup
# -----------------------------------------------------------------------------
def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


seed_everything(42)

utils.setup_logging()
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# CLI Arguments
# -----------------------------------------------------------------------------
CLI_ARGS: dict = utils.parse_args()
_MODEL_CHECKPOINT = CLI_ARGS.get("checkpoint")

# -----------------------------------------------------------------------------
# DEVICE
# -----------------------------------------------------------------------------
_USE_CPU = False
if torch.cuda.is_available() and not _USE_CPU:
    _DEVICE = torch.device("cuda")
    _CUDA_DEVICE = "cuda"
else:
    _DEVICE = torch.device("cpu")
    _CUDA_DEVICE = "cpu"

logger.info(f"Using device: {_DEVICE}")

# -----------------------------------------------------------------------------
# 🔹 Save predictions function supporting multiple deterministic draws
# -----------------------------------------------------------------------------
def safe_prediction_results(
    tasks: List[BaseTask],
    model: LightningModel,
    device: torch.device,
    CFG: dict,
    timestamp: str,
    tag: str = "",
    draw_idx: int = 0,
    base_seed: int = 42
) -> None:

    version = _extract_version_from_string(_MODEL_CHECKPOINT)
    save_path = f"assets/predictions/{version}/{timestamp}/{tag}/draw_{draw_idx}/"
    os.makedirs(save_path, exist_ok=True)

    model = model.to(device)
    model.eval()

    # Determine seed offset for deterministic draws
    if tag == "seen":
        offset = 0
    elif tag == "unseen":
        offset = 1000
    else:
        offset = 2000
    run_seed = base_seed + offset + draw_idx

    for task_index in tqdm(range(len(tasks))):
        logger.info(
            f"Saving predictions for task #{task_index}: {tasks[task_index].get_task_name()} (draw {draw_idx}) ..."
        )

        # 🔹 Reset RNG before creating sampler
        seed_everything(run_seed)

        task_dataset = NeuralizerGeneralDataset(
            tasks=[tasks[task_index]],
            **CFG["training"]["dataset"]
        )

        sampler = WeightedTaskSampler(task_dataset)

        # 🔹 Deterministic DataLoader
        custom_task_dataloader = DataLoader(
            dataset=task_dataset,
            sampler=sampler,
            batch_size=CFG["training"]["dataloader"]["batch_size"],
            num_workers=0  # critical for reproducibility
        )

        # 🔹 Reset RNG before sampling batch
        seed_everything(run_seed)
        x, y, ctx_in, ctx_out, lossfun, class_names = next(iter(custom_task_dataloader))

        x = x.to(device)
        ctx_in = ctx_in.to(device)
        ctx_out = ctx_out.to(device)

        with torch.inference_mode():
            y_pred = model(x, ctx_in, ctx_out)

        assert y_pred.dtype == torch.float32
        assert y_pred.max() <= 1 and y_pred.min() >= 0

        suffix_fig = f"_{task_index}_{tasks[task_index].get_task_name()}"

        plot_prediction_batch(
            x,
            y,
            y_pred,
            ctx_out,
            class_names,
            savefig=True,
            save_path=save_path,
            suffix_fig=suffix_fig,
        )

    logger.info(
        f"+++ Finished saving predictions for tag={tag}, draw={draw_idx} +++"
    )

# -----------------------------------------------------------------------------
# 🔹 Main pipeline
# -----------------------------------------------------------------------------
def load_and_safe_predictions_results() -> None:

    CLI_ARGS: dict = utils.parse_args()
    CFG: dict = utils.load_config(CLI_ARGS.get("config"))

    # Load metadata
    dataloader.load_data(CFG)
    metadata_df: pd.DataFrame = dataloader.get_metadata_df(
        CFG["dataloader"]["metadata_location"]
    )

    # Get tasks
    tasks: List[BaseTask] = taskloader.get_tasks(
        df=metadata_df,
        CFG=CFG["tasks"],
    )

    # Load trained model
    model: LightningModel = training.load_model(
        CFG=CFG,
        path_to_checkpoint=_MODEL_CHECKPOINT,
        cuda_device=_CUDA_DEVICE,
        use_cpu=_USE_CPU,
    )

    # Test tasks
    test_tasks: List[BaseTask] = taskloader.get_test_tasks(
        df=metadata_df,
        CFG=CFG["tasks"],
    )

    timestamp = get_date_and_time_as_string()

    # Generate 3 deterministic draws per dataset type
    for draw_idx in range(3):
        # Seen tasks
        safe_prediction_results(
            tasks=tasks,
            model=model,
            device=_DEVICE,
            CFG=CFG,
            timestamp=timestamp,
            tag="seen",
            draw_idx=draw_idx,
            base_seed=42
        )

        # Unseen tasks
        safe_prediction_results(
            tasks=test_tasks,
            model=model,
            device=_DEVICE,
            CFG=CFG,
            timestamp=timestamp,
            tag="unseen",
            draw_idx=draw_idx,
            base_seed=42
        )

# -----------------------------------------------------------------------------
# 🔹 Utility
# -----------------------------------------------------------------------------
def _extract_version_from_string(path_to_ckpt: str) -> str:
    match = re.search(r"version_\d+", path_to_ckpt)
    return match.group(0) if match else "Unknown Version"

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    load_and_safe_predictions_results()
