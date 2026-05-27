"""
Model by Steffen Czolbe and Adrian V. Dalca
Neuralizer: General Neuroimage Analysis without Re-Training, 2023
See CITATION.md in `src/neuralizer/
"""

import torch
import pytorch_lightning as pl

from typing import Literal, Optional

from src.train.metrics import compute_metrics

from .models.pairwise_conv_avg_model import PairwiseConvAvgModel
from .util.shapecheck import ShapeChecker
from .models.task_specific import get_loss_fn




class LightningModel(pl.LightningModule):
    def __init__(self, hparams, learning_rate, batch_size, test_loss_suffix: Optional[str] = None,cli_args=None):
        super().__init__()
        # save hparams / load hparams
        self.save_hyperparameters(hparams)

        # build model
        self.net = PairwiseConvAvgModel(
            dim=2 if self.hparams.data_slice_only else 3,
            stages=self.hparams.nb_levels,
            in_channels=3,
            out_channels=3,
            inner_channels=self.hparams.nb_inner_channels,
            conv_layers_per_stage=self.hparams.nb_conv_layers_per_stage,
        )

        self.test_loss_suffix = test_loss_suffix
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        # pick loss function from CLI only
        if cli_args is None:
            self.loss_name = "mse"
            lam = 0.5
            window_size = 11
        else:
            self.loss_name = cli_args.get("loss_name")
            lam = cli_args.get("lam", 0.5)
            window_size = cli_args.get("window_size", 11)

        self.loss_fn = get_loss_fn(self.loss_name, lam=lam, window_size=window_size)

    def forward(self, target_in, context_in, context_out):
        sc = ShapeChecker()
        sc.check(target_in, "B C H W", C=3, H=256, W=256)
        sc.check(context_in, "B L C H W")
        sc.check(context_out, "B L C H W")

        # Run network and apply sigmoid to logits
        logits = self.net(context_in, context_out, target_in)
        y_pred = torch.sigmoid(logits)
        sc.check(y_pred, "B C H W")

        return y_pred

    def training_step(self, batch, batch_idx):
        target_in, y, context_in, context_out, lossfun, _ = batch
        y_pred = self.forward(target_in, context_in, context_out)
        if self.loss_name.lower() == "task_specific":
            loss = self.loss_fn(y, y_pred, lossfun)
        else:
            loss = self.loss_fn(y, y_pred)
        self.log(
            "train_loss",
            value=loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=self.batch_size,
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        # Uncomment the following line to also log val/metrics, which makes training a lot longer ...
        self._shared_metric_step(batch, batch_idx, prefix="val")
        self.log_dict(
            {"val_loss": loss}, prog_bar=True, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size
        )

    def test_step(self, batch, batch_idx, prefix: Optional[str] = None):
        loss = self._shared_eval_step(batch, batch_idx)
        self._shared_metric_step(batch, batch_idx, prefix="test")
        if self.test_loss_suffix is None:
            loss_name = "test_loss"  # Will log loss as test_loss for seen tasks
        else:
            loss_name = f"test_loss_{self.test_loss_suffix}"  # Will log loss as test_loss_unseen
        self.log_dict(
            {loss_name: loss},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=self.batch_size,
        )

    def _shared_eval_step(self, batch, batch_idx):
        target_in, y, context_in, context_out, lossfun, _= batch
        y_pred = self.forward(target_in, context_in, context_out)
        if self.loss_name.lower() == "task_specific":
            loss = self.loss_fn(y, y_pred, lossfun)
        else:
            loss = self.loss_fn(y, y_pred)
        return loss

    def _shared_metric_step(self, batch, batch_idx, prefix: Literal["val", "test"]):
        target_in, y, context_in, context_out, lossfun, class_names = batch
        y_pred = self.forward(target_in, context_in, context_out)

        # Iterate over batch and compute and log metric based on what kind of task / image is sampled
        for b_idx in range(y.shape[0]):
            task_name = class_names[b_idx]
            metrics = compute_metrics(
                y_true=y[b_idx],
                y_pred=y_pred[b_idx],
                class_name=class_names[b_idx],
                device=self.device,
            )
            if metrics is None:
               print(f"Warning: compute_metrics returned None for task {task_name} at batch idx {b_idx}")
            # Append prefix and task_name to metrics keys, e.g. "mse" -> "val/GaussianDeblurring/mse"
            metrics = {f"{prefix}/{task_name}/{k}": v for k, v in metrics.items()}
            
            
            self.log_dict(
                metrics, prog_bar=False, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

        lr_scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=6),
            "interval": "epoch",
            "monitor": "val_loss",
        }

        optimization_dictionary = {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
        return optimization_dictionary
