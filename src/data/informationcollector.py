"""
Module, that contains function that extract various metadata from image given path, or other information.
Metadata includes, for example, the dataset name, shape and mode of the image, task name, ...
"""

import logging
import numpy as np

from pathlib import Path
from PIL import Image
from typing import Tuple, List, Union, Optional

from src.data.segmentation_mappings import SegmentationMappings

logger = logging.getLogger(__name__)


def get_shape_of_image(path: str) -> Tuple[int, int, str]:
    """
    Returns width, height and mode (e.g. "RGB") of the given input image
    To read more about modes, see: https://pillow.readthedocs.io/en/stable/handbook/concepts.html

    :param path: str -- Path to the image
    :return: width, height, mode: Tuple[int, int, str]
    """
    with Image.open(path) as img:
        width, height = img.size
        mode = img.mode

        return width, height, mode


def get_dataset_name_based_on_path(supported_datasets: List[str], path: Union[Path, str]) -> str:
    """Function that maps the image (path) to a known Dataset

    :param supported_datasets: List[str] -- List of supported datasets (e.g. ["OCTDL", "KERMANY", ...]
    :param path: Union[Path, str] -- Path to the image
    :return: str -- Name of the Dataset, or 'UNKNOWN DATASET', if image cannot be mapped
    """
    for dataset in supported_datasets:
        if dataset in str(path):
            return dataset

    logger.info(f"Cannot map image '{path}' to a dataset!")
    return "UNKNOWN DATASET"


def get_task_based_on_dataset(dataset_name: str) -> str:
    """Function that maps a dataset to its task

    :param dataset_name: str -- Name of the dataset, e.g. "KERMANY"
    :return: task: str -- Task of the dataset, e.g. "classification"
    """
    task_mapping = {
        "DUKE": "segmentation",
        "RETOUCH": "segmentation",
        "RETOUCH_OCT": "segmentation",
        "KERMANY": "classification",
        "OCTDL": "classification",
        "UMN": "segmentation",
    }

    return task_mapping.get(dataset_name, "UNKNOWN TASK")


def get_ground_truth_based_on_dataset(dataset: str, path: Union[str, Path]) -> Tuple[str, str]:
    """
    Function, that returns the label and segmentation mask of the image.
    If dataset is a classification dataset, set segmentation mask to None.
    If dataset is a segmentation dataset, extract labels from mask via _get_segmentation_mask_labels().

    :param dataset: str -- Name of the dataset. Is used to differentiate between classification and segmentation tasks
    :param path: Union[str, Path] -- Path to the image / file
    :return: label, segmentation_mask: Tuple[str, str]
    """
    if isinstance(path, str):
        path = Path(path)

    # Get ground truth from KERMANY and OCTDL; Ground truth is the direct parent folder
    if dataset in {"KERMANY", "OCTDL"}:
        label = Path(path).parent.stem
        segmentation_mask = None

        # Make KERMANY and OCTDL label of 'NORMAL' common
        if label == "NO":
            label = "NORMAL"

    # Get ground truth from RETOUCH and DUKE; Ground truth is segmentation mask
    elif dataset in {"RETOUCH", "RETOUCH_OCT", "DUKE", "UMN"}:
        parts = list(path.parts)
        parts[-2] = "labels"

        segmentation_mask = str(Path(*parts))
        # Check if label exists, because some test images have no provided labels
        if Path(segmentation_mask).exists():
            # Extract labels (e.g. IRF, ERF, PED, FLUID)
            label = _get_segmentation_mask_labels(segmentation_mask)
        else:
            segmentation_mask = None
            label = None

    # For UNKNOWN datasets
    else:
        label = "UNKNOWN LABEL"
        segmentation_mask = "UNKNOWN MASK"

    return label, segmentation_mask


def _get_segmentation_mask_labels(path: Union[Path, str]) -> Optional[Union[str, List[str]]]:
    """
    Function, that returns the labels of a segmentation mask.
    Labels are None, if no classes were found, or a list, e.g. ["IRF", "SRF"]. If exactly one class, returns its str.
    Function first loads images as numpy array and searches for each unique pixels values.
    Pixel values are then mapped to the corresponding classes,
    e.g. RED (255, 0, 0) for Intra-retinal Fluid (IRF).

    :param path: Union[Path, str] -- Path to the segmentation mask
    :return: Optional[List[str]] -- Found classes; None, str (if 1 class) or a list of strings containing the classes.
    """
    mappings = SegmentationMappings()

    img = Image.open(path)
    img_array = np.array(img)

    # Get unique pixels values, e.g. (0, 0, 0) for black pixels
    unique_pixels = np.unique(img_array.reshape(-1, img_array.shape[2]), axis=0)

    found_classes: List[str] = []
    for color in unique_pixels:
        # Skip Background Pixels, i.e. no class (usually black or dark-grey)
        if (color == mappings.BACKGROUND_COLOR).all(axis=0).any():
            continue
        elif (color == mappings.BLACK).all(axis=0).any():
            continue
        else:
            class_number: int = mappings.SEG_COLOR_MAPPING[tuple(color)]
            class_name: str = mappings.SEG_CLASS_MAPPING[class_number]
            found_classes.append(class_name)

    if len(found_classes) == 0:
        return None
    elif len(found_classes) == 1:
        return found_classes[0]
    else:
        return found_classes
