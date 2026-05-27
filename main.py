"""
Main pipeline of the Neuralizer Model, or this project in general.
Handles all the pipelines steps, i.e.:
- CLI args parsing and config file parsing
- Metadata creation (Image collection)
- Task creation
- DataLoader object creation (train / val / test)
- Fitting (with Tensorboard Logging)

Run via CUDA_VISIBLE_DEVICES=x python3 main.py, where x specifies the GPU ID to use.
CLI Args:
    --fit-neuralizer: Fits Neuralizer
    --fit-single-task-models: Fits Single-Task Models

Default Args:
    --config: str -- Path to the config file (default: "configs/config.yaml")

Example Calls:
CUDA_VISIBLE_DEVICES=x python3 main.py --fit-neuralizer
"""

import logging
import pandas as pd
import pytorch_lightning as pl
import os

import src.utils.utils as utils
import src.data.dataloader as dataloader
import src.tasks.taskloader as taskloader
import src.train.training as training

from typing import List, Optional
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from src.tasks.base_task import BaseTask
from src.neuralizer.lightning_model import LightningModel

utils.setup_logging()
logger = logging.getLogger(__name__)



def main():
    # Parse CLI arguments and config file as dict
    CLI_ARGS: dict = utils.parse_args()
    CFG: dict = utils.load_config(CLI_ARGS.get("config"))

    # Get params which model kind to fit
    fit_neuralizer: bool = CLI_ARGS.get("fit_neuralizer")
    fit_single_task_models: bool = CLI_ARGS.get("fit_single_task_models")

    # Create or get metadata.csv, i.e. load all images and preprocess
    dataloader.load_data(CFG)
    metadata_df: pd.DataFrame = dataloader.get_metadata_df(CFG["dataloader"]["metadata_location"])
    

    # Get training tasks (seen)
    tasks: List[BaseTask] = taskloader.get_tasks(df=metadata_df, CFG=CFG["tasks"])

    # Get test tasks (tasks unseen during training)
    test_tasks: List[BaseTask] = taskloader.get_test_tasks(
        df=metadata_df, CFG=CFG["tasks"]
    )
    
    # Define callbacks to give to Pytorch Lightning Trainer in specific fitting settings
    callbacks = [EarlyStopping(**CFG["training"]["early_stopping"]), LearningRateMonitor(logging_interval="epoch")]

    # Fit Neuralizer trained on all tasks, evaluate on them and also evaluate on test tasks (unseen)
    if fit_neuralizer:
        _fit_neuralizer_on_all_tasks(CFG, tasks, test_tasks, callbacks, CLI_ARGS)
    # Fit Neuralizer on Single Tasks (seen+unseen) and evaluate on them
    elif fit_single_task_models:
        _fit_neuralizer_on_single_tasks(CFG, tasks, test_tasks, callbacks, CLI_ARGS)
    else:
        logger.info("+++ No fitting instruction in CLI args found +++")


def _fit_neuralizer_on_all_tasks(   
    CFG: dict,
    tasks: List[BaseTask],
    test_tasks: List[BaseTask],
    callbacks: List,
    CLI_ARGS: dict=None, 
) -> None:
    """Model is trained on all tasks and tested on tasks seen, as well as tested on tasks unseen.

    :param CFG: dict -- Parsed config file containing all hyperparameters
    :param tasks: List[BaseTask] -- List of initialized tasks to be seen
    :param test_tasks: List[BaseTask] -- List of initialized tasks to be unseen
    :param callbacks: List -- List of Callbacks, e.g. EarlyStopping(.)
    :return: None
    """
    assert CFG["training"]["tensorboard_logger"]["name"] == "GeneralNeuralizer", "Use 'GeneralNeuralizer' as name in config."
    # Get model
    checkpoint_path = CLI_ARGS.get("checkpoint")

    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        model: LightningModel = LightningModel.load_from_checkpoint(checkpoint_path,
            learning_rate=CFG["training"]["model_instantiation"]["learning_rate"],
            batch_size=CFG["training"]["dataloader"]["batch_size"],
            strict=False,
            )
    else:
        logger.info("No checkpoint found, initializing fresh model.")
        model: LightningModel = training.get_model(CFG=CFG,CLI_ARGS=CLI_ARGS)

    
    
    # Get train, validation and test set as DataLoader objects
    train_dataloader, val_dataloader, test_dataloader = training.get_dataloaders(CFG=CFG, tasks=tasks)

    # Get test-tasks as Dataloader
    _, _, test_unseen_dataloader = training.get_dataloaders(CFG=CFG, tasks=test_tasks)

    # Fit model and return fitted Trainer object
    trainer: pl.Trainer = training.fit_model(
        CFG=CFG,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        callbacks=callbacks,
    )

    # Test generalization ability to unseen tasks
    model.test_loss_suffix = "unseen"
    trainer.test(model=model, dataloaders=test_unseen_dataloader, verbose=True)


def _fit_neuralizer_on_single_tasks(
    CFG: dict,
    tasks: List[BaseTask],
    test_tasks: List[BaseTask],
    callbacks: List,
    CLI_ARGS: dict=None,
) -> None:
    """Model is trained on one task and tested on that task again with 60:20:20 split.
    Repeat for all tasks, therefore len(tasks) + len(test_tasks) models are trained and evaluated

    :param CFG: dict -- Parsed config file containing all hyperparameters
    :param tasks: List[BaseTask]  -- List of initialized tasks to be seen
    :param test_tasks: List[BaseTask]  -- List of initialized tasks to be seen (single-tasks models are trained on all)
    :param callbacks: List -- List of Callbacks, e.g. EarlyStopping(.)
    :return: None
    """
    assert CFG["training"]["tensorboard_logger"]["name"] == "GeneralNeuralizer", "Use 'GeneralNeuralizer' as name in config."
    # Combine task, as single models are trained on all available tasks
    tasks_combined = tasks + test_tasks

    for task in tasks_combined:
        task_name: str = task.get_task_name()
        logger.info(f"+++ Fitting Single Model for Task '{task_name}' +++")

        # Instantiate and get model
        model: LightningModel = training.get_model(CFG=CFG,CLI_ARGS=CLI_ARGS)

        # Get train, validation and test set as DataLoader objects
        train_dataloader, val_dataloader, test_dataloader = training.get_dataloaders(CFG=CFG, tasks=[task])

        # Fit model and return fitted Trainer object
        trainer: pl.Trainer = training.fit_model(
            CFG=CFG,
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            callbacks=callbacks,
            task_name=task_name,
        )


if __name__ == "__main__":
    main()
