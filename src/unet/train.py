"""
Module handling the UNet architecture using the pytorch-lightning framework and modelling.

Run from root via `CUDA_VISIBLE_DEVICES=2 python src/unet/train.py`
Open tensorboard via `tensorboard --logdir=logs/tensorboard_logs --port=8008`
"""

import logging
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from typing import Tuple, Dict
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

from src.unet.unet_pl import UNetPL
from src.unet.dataset import OCTDataset
from src.data.dataloader import get_metadata_df
from src.utils.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

_HP: Dict[str, any] = {
    "lr": 1e-4,
    "n_epochs": 64,
    "n_output_channels": 1,
    "device_number": 2,
    "batch_size": 16,
    "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
    "enable_checkpoints": False,
    "log_every_n_steps": 10,
    "reduce_lr_patience": 3,
    "early_stopping_patience": 5,
    "callback_monitor": "val_loss",
    "resize_shape": (192, 192),
    "pl_seed_all": 42,
    "num_workers": 4,
    "p_horizontal_flip": 0.5,
    "p_random_sharpness": 0.5,
    "degree_random_rotation": 10,
    "sharpness_factor": 2,
    "color_jitter_brightness": 0.2,
    "color_jitter_contrast": 0.2,
}


def main():
    pl.seed_everything(seed=_HP["pl_seed_all"], workers=True)

    # Instantiate UNet model
    model = UNetPL(
        n_output_channels=_HP["n_output_channels"],
        bilinear=False,
        learning_rate=_HP["lr"],
        reduce_lr_patience=_HP["reduce_lr_patience"],
        callback_monitor=_HP["callback_monitor"],
    )

    # Get data and do random 60%, 20%, 20% split
    train, validate, test = _get_and_filter_data_splits()

    # Define augmentations to enrich images
    augmented_transforms = _get_augmentation_transformations()

    # Define datasets
    train_dataset = OCTDataset(train, resize_shape=_HP["resize_shape"], augmented_transforms=augmented_transforms)
    val_dataset = OCTDataset(validate, resize_shape=_HP["resize_shape"])
    test_dataset = OCTDataset(test, resize_shape=_HP["resize_shape"])

    # Define dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=_HP["batch_size"],
        shuffle=True,
        num_workers=_HP["num_workers"],
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=_HP["batch_size"], num_workers=_HP["num_workers"], persistent_workers=True
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=_HP["batch_size"], num_workers=_HP["num_workers"], persistent_workers=True
    )

    n_train_dataset, n_val_dataset, n_test_dataset = len(train_dataset), len(val_dataset), len(test_dataset)
    logger.info(f"Length of train dataset: {n_train_dataset}; Number of batches: {len(train_dataloader)}")
    logger.info(f"Length of val dataset: {n_val_dataset}; Number of batches: {len(val_dataloader)}")
    logger.info(f"Length of test dataset: {n_test_dataset}; Number of batches: {len(test_dataloader)}")

    # Instantiate logger and log params
    tb_logger = TensorBoardLogger(save_dir="logs/tensorboard_logs", name="UNetPL")
    tb_logger.log_hyperparams(_HP)
    tb_logger.log_hyperparams(
        {"n_train_dataset": n_train_dataset, "n_val_dataset": n_val_dataset, "n_test_dataset": n_test_dataset}
    )
    tb_logger.log_graph(model)

    # Fit model
    trainer = pl.Trainer(
        accelerator=_HP["accelerator"],
        max_epochs=_HP["n_epochs"],
        logger=tb_logger,
        log_every_n_steps=_HP["log_every_n_steps"],
        default_root_dir="logs/",
        enable_checkpointing=_HP["enable_checkpoints"],
        callbacks=[
            EarlyStopping(monitor=_HP["callback_monitor"], mode="min", patience=_HP["early_stopping_patience"]),
            LearningRateMonitor(logging_interval="epoch"),
        ],
    )
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Test (and log) on test set
    trainer.test(model=model, dataloaders=test_dataloader, verbose=True)


def _get_augmentation_transformations() -> v2.Compose:
    """Simply returns augmentation transformation pipeline
    :return: v2.Compose -- Sequence of transformations
    """
    return v2.Compose(
        [
            v2.RandomHorizontalFlip(p=_HP["p_horizontal_flip"]),
            v2.RandomRotation(degrees=_HP["degree_random_rotation"]),
            v2.RandomAdjustSharpness(sharpness_factor=_HP["sharpness_factor"], p=_HP["p_random_sharpness"]),
        ]
    )


def _get_and_filter_data_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Utility function to get train, validation and test data of the UMN segmentation dataset using the metadata df
    :return: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] -- train, validation and test set
    """
    df = get_metadata_df(path="data/metadata.csv")
    df = df[df["dataset"] == "UMN"]

    train, validate, test = np.split(df.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))])
    return train, validate, test


if __name__ == "__main__":
    main()
