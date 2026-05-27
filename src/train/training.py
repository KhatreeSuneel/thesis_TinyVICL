"""
Module handling the Training Setup and Fitting of various models.
"""

import torch
import os
import logging
import copy
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import json

from typing import List, Tuple, Optional, Union
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.tasks.base_task import BaseTask
from src.dataset.general_dataset import NeuralizerGeneralDataset
from src.dataset.sampler import WeightedTaskSampler
from src.neuralizer.lightning_model import LightningModel
from src.utils.tasks_utils import validate_test_task_samples_are_not_in_train_tasks    



logger = logging.getLogger(__name__)


def fit_model(
    CFG: dict,
    model: LightningModel,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    callbacks: Optional[List] = None,
    task_name: Optional[str] = None,   
) -> pl.Trainer:
    """Fits the model with the provided hyperparams specified in CFG and the DataLoaders objects.

    :param CFG: dict -- Configuration file
    :param model: Union[LightningModel, Retinalizer] -- Model returned from .get_model(), or .get_retinalizer_model()
    :param train_dataloader: DataLoader -- Training data
    :param val_dataloader: DataLoader -- Validation data
    :param test_dataloader: DataLoader -- Test data
    :param callbacks: Optional[List] -- List of Callbacks to give PL Trainer
    :param task_name: Optional[str] -- Name of the task; only relevant for Single Task Model Training
    :return: pl.Trainer
    """
    # Instantiate logger and log params
    save_dir = CFG["training"]["tensorboard_logger"]["save_dir"]
    name = CFG["training"]["tensorboard_logger"]["name"]

    # If a task_name is provided (in single task training), then we log under /OCTNeuralizer-Single/<TASK>
    if task_name is not None:
        name += "-Single/" + task_name


    tb_logger = TensorBoardLogger(save_dir=save_dir, name=name)
    tb_logger.log_hyperparams(CFG)
    
    # Checkpoint callback pointing to logger folder
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        save_last=True,
    )

    # Metrics callback
    class SaveMetricsCallback(pl.Callback):
        def on_train_end(self, trainer, pl_module):
            metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
            save_dir = os.path.join(trainer.logger.log_dir, "checkpoints")
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "final_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)

    # Append these to the existing callbacks
    if callbacks is None:
        callbacks = []
    callbacks += [checkpoint_callback, SaveMetricsCallback()]

    # Fit model
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=tb_logger,
        callbacks=callbacks,
        fast_dev_run=False,
        **CFG["training"]["pl_trainer"],
    )

    logger.info(f"Fitting model for {trainer.max_epochs} epochs ...")
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Test (and log) on test set
    logger.info("Testing model on test dataloader ...")
    trainer.test(model=model, dataloaders=test_dataloader, verbose=True)

    return trainer


def get_model(CFG: dict,CLI_ARGS: dict = None) -> LightningModel:
    """Returns the instantiated LightningModel, i.e. the Neuralizer model

    :param CFG: dict -- Configuration file
    :return: LightningModel
    """
    model = LightningModel(
        CFG["training"]["model_instantiation"],
        learning_rate=CFG["training"]["model_instantiation"]["learning_rate"],
        batch_size=CFG["training"]["dataloader"]["batch_size"],
        cli_args=CLI_ARGS,
    )
    return model




def get_dataloaders(
    CFG: dict, tasks: List[BaseTask], test_mode: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Defines and returns train, validation and test dataloaders

    :param CFG: dict -- Configuration file
    :param tasks: List[BaseTask] -- List of BaseTasks to consider during training / val / testing
    :param vendor: Optional[str] -- Only relevant for RETOUCH Domain Adaption Models
    :param test_mode: bool -- Whether to activate test mode in Test Set / Sampler, or not. If activated,
                              no weighted sampling occurs, i.e. the test dataset contains all test samples returned
                              in sequential order; Also no stochasticity, i.e. each sample has its own context.
                              If set to False, it might be possible that a test sample is drawn multiple times, while
                              some test samples are not drawn at all. Should be set to True in evaluation scripts,
                              to ensure test samples are exactly once deterministically evaluated in sequential order.
    :return: Tuple[DataLoader, DataLoader] -- Train and validation Dataloader
    """
    
    # Split tasks equally (Setting 4: All dataset are present during training)
    tasks_train, tasks_val, tasks_test = _split_tasks_equally(tasks, CFG=CFG)
    

    # Define train dataset and passing simple data augmentation
    train_dataset = NeuralizerGeneralDataset(
        tasks=tasks_train, **CFG["training"]["dataset"]
    )
    sampler_train = WeightedTaskSampler(train_dataset)

    # Define val dataset
    val_dataset = NeuralizerGeneralDataset(tasks=tasks_val, **CFG["training"]["dataset"])
    sampler_val = WeightedTaskSampler(val_dataset)

    # Define test dataset
    test_dataset = NeuralizerGeneralDataset(tasks=tasks_test, **CFG["training"]["dataset"], test_mode=test_mode)
    sampler_test = WeightedTaskSampler(test_dataset, test_mode=test_mode)

    # Define dataloaders
    train_dataloader = DataLoader(dataset=train_dataset, sampler=sampler_train, **CFG["training"]["dataloader"])
    val_dataloader = DataLoader(dataset=val_dataset, sampler=sampler_val, **CFG["training"]["dataloader"])
    test_dataloader = DataLoader(dataset=test_dataset, sampler=sampler_test, **CFG["training"]["dataloader"])

    n_train_dataset, n_val_dataset, n_test_dataset = len(train_dataset), len(val_dataset), len(test_dataset)
    logger.info(f"Length of train dataset: {n_train_dataset}; Number of batches: {len(train_dataloader)}")
    logger.info(f"Length of val dataset: {n_val_dataset}; Number of batches: {len(val_dataloader)}")
    logger.info(f"Length of test dataset: {n_test_dataset}; Number of batches: {len(test_dataloader)}")

    return train_dataloader, val_dataloader, test_dataloader


def _split_tasks_equally(tasks: List[BaseTask], CFG: dict) -> Tuple[List[BaseTask], List[BaseTask], List[BaseTask]]:
    """Split tasks according to setting 4, i.e. all datasets/tasks are present in train/val/test.

    If in config "filter_segmentation_test_tasks" is set to True:
    Then we only include non-healthy scans in the test tasks (segmentation).
    We do this, because we want to evaluate only on non-empty-segmentation masks, i.e.
    where there are some classes (beside the background) in the segmentation mask.

    :param tasks: List[BaseTask] -- List of Tasks to split in train/val/test according to setting 4
    :param CFG: dict -- Parsed config file
    :return: Tuple[List[BaseTask], List[BaseTask], List[BaseTask]] -- Tasks split in 60:20:20 ratio
    """
    logger.info("Splitting Tasks equally with 60:20:20 proportion ...")

    tasks_train, tasks_val, tasks_test = [], [], []
    for task in tasks:
        # Get 100% dataframe from task
        df = task.get_data()
        task_name = task.get_task_name()

        # Split in 60% train, 20% val, 20% test
        train, validate, test = np.split(df.sample(frac=1, random_state=42), [int(0.6 * len(df)), int(0.8 * len(df))])
        # Copy original task object three times for train, val and test
        task_train, task_val, task_test = copy.deepcopy(task), copy.deepcopy(task), copy.deepcopy(task)

        # Overwrite data attribute with new dataset proportion
        task_train.data = pd.DataFrame(train)
        task_val.data = pd.DataFrame(validate)
        task_test.data = pd.DataFrame(test)

        # Append to list
        tasks_train.append(task_train)
        tasks_val.append(task_val)
        tasks_test.append(task_test)

    logger.info("Validating if Test Task Samples are in no Train Task Partition ...")
    for test_task in tasks_test:
        validate_test_task_samples_are_not_in_train_tasks(test_task, tasks_train)

    return tasks_train, tasks_val, tasks_test


def get_augmentation_transformations(CFG: dict) -> v2.Compose:
    """Simply returns simple augmentation transformation pipeline applied for the train set

    :param CFG: dict -- Parsed config file
    :return: v2.Compose -- Sequence of transformations
    """
    return v2.Compose([v2.RandomHorizontalFlip(p=CFG["training"]["data_augmentation"]["p_horizontal_flip"])])


def load_model(
    CFG: dict, path_to_checkpoint: str, cuda_device: Union[str, torch.device], use_cpu: bool
) -> LightningModel:
    """Simply loads trained model from checkpoint

    :param CFG: dict -- Parsed config file
    :param path_to_checkpoint: str -- Path to the checkpoint
    :param cuda_device: [str, torch.device] -- Cuda device
    :param use_cpu: bool -- Whether to use CPU or GPU for inference
    :return: model: LightningModel
    """
    # Load fitted model
    model: LightningModel = get_model(CFG=CFG)
    checkpoint = torch.load(path_to_checkpoint, map_location="cpu" if use_cpu else cuda_device)
    model.load_state_dict(checkpoint["state_dict"])
    return model
