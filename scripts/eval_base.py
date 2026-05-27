"""
Evaluation object for loading trained models in a controlled way.
Supports:
- Multiple architectures (via different CFGs)
- Random baseline (checkpoint_path=None)
"""

import logging
import torch

import src.train.training as training

from pathlib import Path
from dataclasses import dataclass, field
from typing import Union, Optional

from src.neuralizer.lightning_model import LightningModel

logger = logging.getLogger(__name__)


# =====================================================
# EVAL OBJECT
# =====================================================

@dataclass()
class EvalObject:
    name: str
    checkpoint_path: Optional[Union[str, Path]]
    device: Union[str, torch.device]
    CFG: dict

    def __post_init__(self):
        self.model = self._load_model()

    # -------------------------------------------------

    def _load_model(self) -> Optional[LightningModel]:
        """
        Loads model from checkpoint using provided CFG.
        Returns None for random baseline.
        """

        if self.checkpoint_path is None:
            logger.info(f"[{self.name}] Random baseline (no checkpoint).")
            return None

        ckpt_path = Path(self.checkpoint_path)

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"[{self.name}] Checkpoint not found at '{ckpt_path}'."
            )

        logger.info(f"[{self.name}] Loading checkpoint: {ckpt_path}")

        model = training.load_model(
            CFG=self.CFG,
            path_to_checkpoint=str(ckpt_path),
            cuda_device=self.device,
            use_cpu=(str(self.device) == "cpu"),
        )

        model.eval()
        model.to(self.device)

        logger.info(f"[{self.name}] Loaded successfully.")

        return model

    # -------------------------------------------------

    def get_model(self) -> Optional[LightningModel]:
        return self.model

    # -------------------------------------------------

    def get_name(self) -> str:
        return self.name
