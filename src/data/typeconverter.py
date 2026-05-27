"""
Module, that handles the image type and mode conversion (i.e. from * to .png type, from L to RGB).
We need this to make sure, that all images (including segmentation masks) are of the same type and mode
The type and mode is specified in the configs/config.yaml (default: '.png' and 'RGB')
"""

import os
import logging

import pandas as pd

from pathlib import Path
from tqdm import tqdm
from PIL import Image

logger = logging.getLogger(__name__)


def convert_imgs_to_type(df: pd.DataFrame, img_type: str = ".png",path_column: str = "path", suffix_column: str = "suffix") -> None:
    """
    Converts all images of the input dataframe to the specified type.
    Assume that the input df has the column "path", which links to the images.

    :param df: pd.DataFrame -- Dataframe containing the paths to all images
    :param img_type: str -- The format to convert the image to (default: '.png')
    :return: None
    """
    if path_column not in df.columns or suffix_column not in df.columns:
        raise ValueError(f"Columns '{path_column}' and/or '{suffix_column}' not found in input df!")


    # Get only rows where image is not of desired type
    df_temp = df[(df[suffix_column] != img_type)]

    logger.info(f"Converting {len(df_temp)} images to desired type '{img_type}' ...")
    for row in tqdm(df_temp.itertuples(), total=len(df_temp)):
        try:
            _convert_img_to_type(path=getattr(row, path_column), img_type=img_type)
        except FileNotFoundError as e:
            logger.exception(f"{e}: Skipping file")


def convert_imgs_to_mode(df: pd.DataFrame, img_mode: str = "RGB", path_column: str = "path", mode_column: str = "mode") -> None:
    """
    Converts all images of the input dataframe to the specified mode.
    Assume that the input df has the column "path", which links to the images.

    :param df: pd.DataFrame -- Dataframe containing the paths to all images
    :param img_mode: str -- The format to convert the image to (default: '.png')
    :return: None
    """
    if path_column not in df.columns or mode_column not in df.columns:
        raise ValueError(f"Columns '{path_column}' and/or '{mode_column}' not found in input df!")

    # Get only rows where image is not of desired type
    df_temp = df[(df[mode_column] != img_mode)]

    logger.info(f"Converting {len(df_temp)} images to desired mode '{img_mode}' ...")
    for row in tqdm(df_temp.itertuples(), total=len(df_temp)):
        try:
            _convert_img_to_mode(path=getattr(row, path_column), img_mode=img_mode)
        except FileNotFoundError as e:
            logger.exception(f"{e}: Skipping file")


def _convert_img_to_type(path: str, img_type: str = ".png") -> None:
    """
    Converts a given image specified by the input parameter path to the specific type (e.g. to PNG format).

    :param path: str -- Path to the image to convert
    :param img_type: str -- The format to convert the image to (default: '.png')
    :return: None
    """
    path: Path = Path(path)
    name: str = path.stem  # Get name of file without suffix, e.g. "vid_vmt_9"
    sfx: str = path.suffix  # Get current suffix, e.g. ".jpeg"
    folder: Path = Path(path).parent  # Get folder the file lies in, e.g. "data/raw/OCTDL/OCTDL/VID"

    # If suffix is not of desired format already: Convert image. Otherwise, skip converting.
    if sfx != img_type:
        logger.debug(f"Converting image '{name}' with suffix '{sfx}' to '{img_type}' ...")
        # Open Image and save it as png file
        img = Image.open(path)
        img.save(os.path.join(folder, name + img_type))

        # Delete old original file, as one file is enough
        Path.unlink(path)


def _convert_img_to_mode(path: str, img_mode: str = "RGB") -> None:
    """
    Converts a given image specified by the input parameter path to the specific mode

    :param path: str -- Path to the image to convert
    :param img_mode: str -- The format to convert the image to (default: '.png')
    :return: None
    """
    path: Path = Path(path)
    name: str = path.stem  # Get name of file without suffix, e.g. "vid_vmt_9"

    logger.debug(f"Converting image '{name}' to '{img_mode}' ...")
    # Open Image and save it as png file
    img = Image.open(path)
    img = img.convert(img_mode)
    # Overwrite image
    img.save(path)
