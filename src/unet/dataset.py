""" Module handling the custom OCT Dataset inheriting from torch.utils.data.Dataset"""

import torch
import pandas as pd
import numpy as np

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2

from typing import Tuple, Optional


class OCTDataset(Dataset):

    def __init__(
        self,
        df: pd.DataFrame,
        resize_shape: Tuple[int, int] = (192, 192),
        augmented_transforms: Optional[v2.Compose] = None,
    ):
        """
        :param df: pd.DataFrame -- Dataframe containing 'path' and 'segmentation_mask' column
        :param resize_shape: Tuple[int, int] -- Shape to resize images to
        :param augmented_transforms: Optional[v2.Compose] -- Transformation used for augmenting the original dataset
        """
        self.df = df
        self.resize_shape = resize_shape
        self.augmented_transforms = augmented_transforms

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Loads one image and label example after minor preprocessing

        :param idx: int -- Index of the provided sample
        :return: image, label: Tuple[Tensor, Tensor] -- Image and corresponding segmentation mask (label)
        """
        path_to_image = self.df["path"].iloc[idx]
        path_to_seg = self.df["segmentation_mask"].iloc[idx]

        # Open and resize the image and label
        image = Image.open(path_to_image).resize(size=self.resize_shape)
        label = Image.open(path_to_seg).resize(size=self.resize_shape)

        # Define base transform that is always performed for the input X
        base_transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        image = base_transforms(image)

        # Apply augmented transformation if provided
        if self.augmented_transforms is not None:
            image, label = self.augmented_transforms(image, label)

        # Convert label to numpy array; Make 1 Channel Label and convert to tensor
        label = np.array(label)
        label = _rgb_to_class_mask(rgb_mask=label)
        label = torch.from_numpy(label.astype(np.uint8)).unsqueeze(0)

        return image, label


def _rgb_to_class_mask(rgb_mask):
    """
    Utility function to convert 3 Channel (RGB) segmentation mask to one channel,
    where each pixel has one unique encoding.

    :param rgb_mask: Input segmentation mask of shape (3, H, W)
    :return: One channel segmentation mask of shape (H, W)
    """
    rgb_to_class_mapping = {(0, 0, 0): 0, (255, 255, 0): 1}
    height, width, _ = rgb_mask.shape
    class_mask = np.zeros((height, width), dtype=np.uint8)

    for rgb, class_index in rgb_to_class_mapping.items():
        match_pixels = (rgb_mask == rgb).all(axis=-1)
        class_mask[match_pixels] = class_index

    return class_mask
