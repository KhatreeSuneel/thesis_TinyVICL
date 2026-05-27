"""Module handling the postprocessing of the predictions"""

import logging

import numpy as np
import torch
import torch.nn.functional as F

from typing import List, Tuple, Literal
from src.utils.colors_utils import decode_color_to_index


logger = logging.getLogger(__name__)


from PIL import Image

def post_process_to_indices(y: torch.Tensor, y_pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts color-coded (RGB) tensors y and y_pred to index maps (class indices per pixel).

    :param y: torch.Tensor (3, H, W) or PIL.Image.Image
    :param y_pred: torch.Tensor (3, H, W) or PIL.Image.Image
    :return: (y_idx_list, y_pred_idx_list) -- flattened lists of class indices
    """

    def to_pil(img):
        if isinstance(img, Image.Image):
            return img
        elif isinstance(img, torch.Tensor):
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            return Image.fromarray(img_np)
        else:
            raise TypeError(f"Unsupported input type: {type(img)}")

    # Convert to PIL
    y_pil = to_pil(y)
    y_pred_pil = to_pil(y_pred)

    # Decode colors → indices
    y_idx = decode_color_to_index(y_pil)
    y_pred_idx = decode_color_to_index(y_pred_pil)

    # Flatten and convert to list
    y_idx_list = y_idx.flatten().tolist()
    y_pred_idx_list = y_pred_idx.flatten().tolist()

    return y_idx_list, y_pred_idx_list




