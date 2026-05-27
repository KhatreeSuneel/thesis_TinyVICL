"""
Standalone script to load a model and save predictions results
from the test dataloader to assets/./predictions

Run from root via
python3 scripts/save_predictions_figures.py --checkpoint path_to_ckpt
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

utils.setup_logging()
logger = logging.getLogger(__name__)

# Path to model checkpoint to load
CLI_ARGS: dict = utils.parse_args()
_MODEL_CHECKPOINT = CLI_ARGS.get("checkpoint")

# Specify cuda device
# Device selection (uses GPU if available)
_USE_CPU = False  # Set False to use GPU

if torch.cuda.is_available() and not _USE_CPU:
    _DEVICE = torch.device("cuda")
    _CUDA_DEVICE = "cuda"
else:
    _DEVICE = torch.device("cpu")
    _CUDA_DEVICE = "cpu"

logger.info(f"Using device: {_DEVICE}")

# Set seeds for reproducing
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True


def safe_prediction_results(
    tasks: List[BaseTask],
    model: LightningModel,
    device: torch.device,
    CFG: dict,
    dataloader: DataLoader,
    timestamp: str,
    tag="",
) -> None:
    """Function to save a sample from the test dataloader to see how good the model performs.

    :param tasks: List[BaseTask] -- List of Tasks
    :param model: LightningModel -- Loaded model
    :param device: torch.device -- CPU or GPU device
    :param CFG: dict -- Parsed config
    :param dataloader: DataLoader -- PyTorch Dataloader
    :param timestamp: str -- Current time and date as string
    :return: None
    """
    # Get version from the checkpoint path
    version = _extract_version_from_string(path_to_ckpt=_MODEL_CHECKPOINT)
    # Define save_path to save results to
    save_path = f"assets/predictions/{version}/{timestamp}/{tag}/"
    os.makedirs(save_path, exist_ok=True)
    
    # Put data and model to device
    model = model.to(device)
    model.eval()

    # Iterate over each task
    for task_index in tqdm(range(len(tasks))):
        logger.info(f"Saving predictions for task #{task_index}: {tasks[task_index].get_task_name()} ...")

        # Extract from the test dataloader the task of interest
        task_dataset = NeuralizerGeneralDataset(tasks=[dataloader.dataset.tasks[task_index]], **CFG["training"]["dataset"])
        sampler = WeightedTaskSampler(task_dataset)
        custom_task_dataloader = DataLoader(dataset=task_dataset, sampler=sampler, **CFG["training"]["dataloader"])
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # To get a random sample from all tasks inside the test_dataloader, iterate over test_dataloader
        # To get a random sample from a specific task, specify the task index and iterate over custom_task_dataloader
        x, y, ctx_in, ctx_out, lossfun, class_names = next(iter(custom_task_dataloader))

        x = x.to(device)
        ctx_in = ctx_in.to(device)
        ctx_out = ctx_out.to(device)

        # Do inference
        with torch.inference_mode():
            y_pred = model(x, ctx_in, ctx_out)

        assert y_pred.dtype == torch.float32 and y_pred.max() <= 1 and y_pred.min() >= 0

        suffix_fig = custom_task_dataloader.dataset.tasks[0].get_task_name()
        suffix_fig = f"_{task_index}_" + suffix_fig

        plot_prediction_batch(
            x, y, y_pred, ctx_out, class_names, savefig=True, save_path=save_path, suffix_fig=suffix_fig
        )

    logger.info("+++ End of 'save_predictions_figures.py' pipeline: Successfully saved images +++")


def load_and_safe_predictions_results() -> None:
    """Pipeline to load model and dataloader and call safe_prediction_results

    :return: None
    """
    # Parse CLI arguments and config file as dict
    CLI_ARGS: dict = utils.parse_args()
    CFG: dict = utils.load_config(CLI_ARGS.get("config"))

    # Create or get metadata.csv, i.e. load all images and preprocess
    dataloader.load_data(CFG)
    metadata_df: pd.DataFrame = dataloader.get_metadata_df(CFG["dataloader"]["metadata_location"])

    # Get tasks
    tasks: List[BaseTask] = taskloader.get_tasks(df=metadata_df,  CFG=CFG["tasks"])

    # Load trained Model
    model: LightningModel = training.load_model(
        CFG=CFG, path_to_checkpoint=_MODEL_CHECKPOINT, cuda_device=_CUDA_DEVICE, use_cpu=_USE_CPU
    )

    # Get train, validation and test set as DataLoader objects
    _, _, test_dataloader = training.get_dataloaders(CFG=CFG, tasks=tasks, test_mode=True)

    # Get test tasks (tasks not seen during training)
    test_tasks: List[BaseTask] = taskloader.get_test_tasks(
        df=metadata_df, CFG=CFG["tasks"]
    )

    # Get test-tasks as Dataloader
    _, _, test_tasks_dataloader = training.get_dataloaders(CFG=CFG, tasks=test_tasks, test_mode=True)

    # Get timestamp to save everything in right folder
    timestamp = get_date_and_time_as_string()

    # Safe predictions results from test tasks seen
    safe_prediction_results(
        tasks=tasks, model=model, device=_DEVICE, CFG=CFG, dataloader=test_dataloader, timestamp=timestamp,tag="seen"
    )

    # Safe predictions results from test tasks unseen
    safe_prediction_results(
        tasks=test_tasks, model=model, device=_DEVICE, CFG=CFG, dataloader=test_tasks_dataloader, timestamp=timestamp,tag='unseen'
    )


def _extract_version_from_string(path_to_ckpt: str) -> str:
    """Extract given version of a path to a ckpt, e.g. 'version_85'

    :param path_to_ckpt: str -- Path to the checkpoint, from where the model is loaded
    :return: version: str -- Version of the model, e.g. 'version_85'
    """
    # Regex to match 'version_' followed by digits
    match = re.search(r"version_\d+", path_to_ckpt)
    return match.group(0) if match else "Unknown Version"


if __name__ == "__main__":
    load_and_safe_predictions_results()
