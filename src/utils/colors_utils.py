"""Module containing utility function for handling segmentation mapping"""

import random
import torch
import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor
from src.data.segmentation_mappings import SegmentationMappings
from typing import List, Tuple, Dict


# def default_color_mapping():    

def colorize_with_default_label(label_pil: Image.Image):
    """
    Convert a grayscale label (index per pixel) to RGB using SegmentationMappings.default_colormap.
    """
    label_np = np.array(label_pil, dtype=np.uint8)  # shape: (H, W)
    if label_np.ndim == 3:
        # Assuming all channels are identical
        label_np = label_np[..., 0]  

    H, W = label_np.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    # Map each index to RGB
    for idx, color in SegmentationMappings.default_colormap.items():
        mask = (label_np == idx)
        rgb[mask] = color

    return Image.fromarray(rgb)

def colorize_with_custom_label(label_pil: Image.Image):
    """
    Convert a grayscale label (index per pixel) to RGB using SegmentationMappings.default_colormap.
    """
    label_np = np.array(label_pil, dtype=np.uint8)  # shape: (H, W)

    H, W = label_np.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    # Map each index to RGB
    for idx, color in SegmentationMappings.default_colormap.items():
        mask = (label_np == idx)
        rgb[mask] = color

    return Image.fromarray(rgb)

def decode_color_to_index(label_pil: Image.Image) -> np.ndarray:
    """
    Convert a color (RGB) segmentation label to a 2D array of class indices
    using SegmentationMappings.default_colormap.

    :param label_pil: PIL.Image.Image -- RGB image (color-coded segmentation)
    :return: np.ndarray -- 2D array (H, W) of class indices
    """
    label_np = np.array(label_pil, dtype=np.uint8)  # (H, W, 3)
    H, W, _ = label_np.shape

    # Precompute inverse colormap (color → class index)
    inverse_colormap = {
        tuple(color): idx for idx, color in SegmentationMappings.default_colormap.items()
    }

    # Flatten for vectorized comparison
    flat_pixels = label_np.reshape(-1, 3)

    # Create an index map filled with -1 initially
    index_map = np.full((flat_pixels.shape[0],), fill_value=-1, dtype=np.int32)

    # Match each unique color in the image
    unique_colors = np.unique(flat_pixels, axis=0)
    for color in unique_colors:
        color_tuple = tuple(color)
        if color_tuple in inverse_colormap:
            idx = inverse_colormap[color_tuple]
            matches = np.all(flat_pixels == color, axis=1)
            index_map[matches] = idx
        else:
            # Optional: handle unknown colors gracefully
            pass

    return index_map.reshape(H, W)


