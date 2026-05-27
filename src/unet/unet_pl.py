"""
Module handling the UNet architecture using the pytorch-lightning framework and modelling.
Gets instantiated in `src/unet/train.py`
"""

import torch
import pytorch_lightning as pl

from torchmetrics import Dice
from torch.nn import functional as F

from src.unet.unet import UNet


class UNetPL(pl.LightningModule):

    def __init__(
        self, n_output_channels=1, bilinear=False, learning_rate=1e-3, reduce_lr_patience=2, callback_monitor="val_loss"
    ):
        super(UNetPL, self).__init__()

        self.model = UNet(n_output_channels=n_output_channels, bilinear=bilinear)
        self.learning_rate = learning_rate
        self.reduce_lr_patience = reduce_lr_patience
        self.callback_monitor = callback_monitor

        self.loss_fn = F.binary_cross_entropy_with_logits  # Combines a Sigmoid layer and BCE Loss
        self.dice = Dice(ignore_index=0, average="micro")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y.to(torch.float64))
        self.log("train_loss", value=loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, dice = self._shared_eval_step(batch, batch_idx)
        self.log_dict(
            {"val_loss": loss, "val_dice": dice},
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )

    def test_step(self, batch, batch_idx):
        loss, dice = self._shared_eval_step(batch, batch_idx)
        self.log_dict(
            {"test_loss": loss, "test_dice": dice},
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0, threshold=0.5):
        x, _ = batch
        logits = self.model(x)
        probs = torch.sigmoid(logits)
        predictions = probs > threshold
        return predictions

    def _shared_eval_step(self, batch, batch_idx, threshold=0.5):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y.to(torch.float64))
        preds = torch.sigmoid(logits) > threshold
        dice = self.dice(preds, y)
        return loss, dice

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, patience=self.reduce_lr_patience
            ),
            "interval": "epoch",
            "monitor": self.callback_monitor,
        }

        optimization_dictionary = {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
        return optimization_dictionary
