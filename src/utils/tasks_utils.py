"""Module containing utility function for tasks"""

import logging
import numpy as np
import pandas as pd

# from src.tasks.base_task import QualityAssuredTask, SemanticallyEnrichedTask, SegmentationMaskTransformationTask
from src.utils.utils import get_tasks_for_segmentation,get_edge_detection_tasks
from src.tasks.base_task import BaseTask

from typing import List

logger = logging.getLogger(__name__)



def is_segmentation_task(class_name: str) -> bool:
    """Checks, whether a class is a segmentation task or not

    :param class_name: str -- Name of the class to check
    :return: bool
    """
    # Check if class name in QualityAssuredTasks and if "segmentation" is in name
    task_list = get_tasks_for_segmentation()
    cond1 = class_name in task_list

    return cond1

def is_edge_detection_task(class_name: str) -> bool:
    """Checks, whether a class is a segmentation task or not

    :param class_name: str -- Name of the class to check
    :return: bool
    """
    # Check if class name in QualityAssuredTasks and if "segmentation" is in name
    task_list = get_edge_detection_tasks()
    cond1 = class_name in task_list

    return cond1


def validate_test_task_samples_are_not_in_train_tasks(test_task: BaseTask, train_tasks: List[BaseTask]):
    """Utility Function to check whether a sample of a test task (identified by its index) is in any train task partition.

    :param test_task: BaseTask -- Test Task to inspect
    :param train_tasks: List[BaseTask] -- List of Train Task to search for
    :return:
    """
    df_test = test_task.get_data()
    for train_task in train_tasks:
        df_train = train_task.get_data()
        assert not df_test.index.isin(
            df_train.index
        ).all(), f"{test_task.get_task_name()} has samples in train task {train_task.get_task_name()}"
