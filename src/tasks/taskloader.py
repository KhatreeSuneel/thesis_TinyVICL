"""
Task loader module for simplified preprocessed tasks:
- Denoising
- Deraining
- Segmentation
Each task is defined using paired image-label data with minimal post-processing.
"""

import pandas as pd
import logging
from typing import List
from src.tasks.base_task import BaseTask
from src.tasks.denoising import DenoisingTask
from src.tasks.deraining import DerainingTask
from src.tasks.segmentation import SegmentationTask
from src.tasks.inpainting import InpaintingTask
from src.tasks.superesolution import SuperresolutionTask
from src.tasks.depthestimation import DepthTask
from src.tasks.edge import EdgeTask


# def get_tasks(df: pd.DataFrame, CFG: dict) -> List[BaseTask]:
#     """
#     Initializes and returns all task objects for training.

#     :param df: pd.DataFrame -- Metadata with image_path, label_path, task
#     :param CFG: dict -- Configuration dictionary with optional task parameters
#     :return: List[BaseTask] -- List of task objects
#     """ 
#     return [
#         DenoisingTask(df, task_name="Denoising", **CFG["denoising"]),
#         DerainingTask(df, task_name="Deraining", **CFG["deraining"]),
#         SegmentationTask(df, task_name="Segmentation", **CFG["segmentation"]),
#         InpaintingTask(df, task_name="Inpainting",**CFG["inpainting"]),
#         SuperresolutionTask(df,task_name="Superresolution",**CFG["superresolution"]),
#         DepthTask(df,task_name="Depthestimation",**CFG["depth_estimation"]),
#         EdgeTask(df,task_name="Edgedetection",**CFG["edge_detection"])
#     ]


# def get_test_tasks(df: pd.DataFrame, CFG: dict) -> List[BaseTask]:
#     """
#     Initializes and returns tasks used for testing/generalization.

#     For now, identical to `get_tasks`, but can customize this to load only unseen/held-out tasks.

#     :param df: pd.DataFrame -- Metadata with image_path, label_path, task
#     :param CFG: dict -- Configuration dictionary with optional task parameters
#     :return: List[BaseTask] -- List of test task objects
#     """
#     return get_tasks(df, CFG)

logger = logging.getLogger(__name__)
def get_tasks(df: pd.DataFrame, CFG: dict) -> List[BaseTask]:
    """
    Initializes and returns all task objects for training.
    Skips tasks with no data instead of raising errors.
    """
    tasks = []
    failed = []

    task_defs = [
        ("Denoising", DenoisingTask, "denoising"),
        ("Deraining", DerainingTask, "deraining"),
        ("Segmentation", SegmentationTask, "segmentation"),
        ("Inpainting", InpaintingTask, "inpainting"),
        ("Superresolution", SuperresolutionTask, "superresolution"),
        ("Depthestimation", DepthTask, "depth_estimation"),
        ("Edgedetection", EdgeTask, "edge_detection"),
    ]

    for task_name, TaskClass, cfg_key in task_defs:
        try:
            task = TaskClass(df, task_name=task_name, **CFG[cfg_key])
            tasks.append(task)
            logger.info(f"[TaskLoader] ✅ Loaded task: {task_name}")
        except ValueError as e:  # e.g. "Data is empty..."
            failed.append(task_name)
            logger.warning(f"[TaskLoader] ⚠️ Skipped task '{task_name}': {e}")
        except Exception as e:
            failed.append(task_name)
            logger.error(f"[TaskLoader] ❌ Failed to init task '{task_name}': {e}", exc_info=True)

    logger.info(f"[TaskLoader] Final Tasks Loaded: {[t.task_name for t in tasks]}")
    if failed:
        logger.info(f"[TaskLoader] Missing/Skipped Tasks: {failed}")

    return tasks


def get_test_tasks(df: pd.DataFrame, CFG: dict) -> List[BaseTask]:
    """
    Initializes and returns tasks used for testing/generalization.
    Skips tasks with no data.
    """
    return get_tasks(df, CFG)
