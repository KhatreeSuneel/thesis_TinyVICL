"""
Module containing utility function relevant for RETOUCH-Domain Adaption Neuralizer Fitting
"""

import re
import copy
import logging

import numpy as np
import pandas as pd

from typing import List, Optional, Tuple

from src.tasks.base_task import BaseTask
from src.utils.tasks_utils import validate_test_task_samples_are_not_in_train_tasks

logger = logging.getLogger(__name__)


def split_tasks_for_domain_adaption(
    CFG: dict, tasks: List[BaseTask], vendor: str
) -> Tuple[List[BaseTask], List[BaseTask], List[BaseTask]]:
    """Splitting function for domain adoption model training
    Will filter RETOUCH Tasks according to the vendor to be left out and adapts the tasks accordingly

    :param CFG: dict
    :param tasks: List[BaseTask]
    :param vendor: str
    :return:
    """

    tasks_train, tasks_val, tasks_test = [], [], []

    for task in tasks:
        # Get 100% dataframe from task
        df = task.get_data()

        # Split in 60% train, 20% val, 20% test
        train, validate, test = np.split(
            df.sample(frac=1, random_state=42), [int(0.6 * len(df)), int(0.8 * len(df))]
        )
        # Copy original task object three times for train, val and test
        task_train, _, _ = copy.deepcopy(task), copy.deepcopy(task), copy.deepcopy(task)

        # Overwrite data attribute with new dataset proportion
        task_train.data = pd.DataFrame(train)
        # task_val.data = pd.DataFrame(validate)
        # task_test.data = pd.DataFrame(test)

        # Append to list
        tasks_train.append(task_train)
        # tasks_val.append(task_val)
        # tasks_test.append(task_test)

    logger.info("Validating if Test Task Samples are in no Train Task Partition ...")
    for test_task in tasks_test:
        validate_test_task_samples_are_not_in_train_tasks(test_task, tasks_train)

    _log_tasks(tasks_train, tasks_val, tasks_test)

    return tasks_train, tasks_val, tasks_test



def _log_tasks(tasks_train: List[BaseTask], tasks_val: List[BaseTask], tasks_test: List[BaseTask]) -> None:
    """Simply utility function to plot the results from the splitting function for manual validation.

    :param tasks_train: List[BaseTask]
    :param tasks_val: List[BaseTask]
    :param tasks_test: List[BaseTask]
    :return:None
    """
    logger.info(f"Tasks included in tasks_train:\n{[task.get_task_name() for task in tasks_train]}")
    logger.info(f"Tasks included in tasks_val:\n{[task.get_task_name() for task in tasks_val]}")
    logger.info(f"Tasks included in tasks_test:\n{[task.get_task_name() for task in tasks_test]}")
