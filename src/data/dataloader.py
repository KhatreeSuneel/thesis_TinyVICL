import os
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict

from PIL import Image
import src.data.typeconverter as typeconverter

logger = logging.getLogger(__name__)


def load_data(CFG: Dict) -> None:
    """
    Executes pipeline:
    1. Collect metadata of all image-label pairs
    2. Convert image/label types and modes if needed
    3. Save metadata
    """
    metadata_path = CFG["dataloader"]["metadata_location"]
    search_dir = CFG["dataloader"]["search_dir"]
    img_type = CFG["dataloader"]["type"]
    img_mode = CFG["dataloader"]["mode"]
    supported_types = CFG["dataloader"]["types"]


    if Path(metadata_path).exists():
        logger.info(f"'{metadata_path}' already exists! Delete it to regenerate.")
        return

    logger.info("Collecting metadata...")
    df = _get_meta_data_with_labels(search_dir, supported_types)
    recollect = False


        # Convert mode
    if _has_faulty_image_mode(df=df, img_mode=img_mode):
        typeconverter.convert_imgs_to_mode(df, img_mode, 
                                            path_column="image_path", mode_column="image_mode")
        typeconverter.convert_imgs_to_mode(df, img_mode, 
                                            path_column="label_path", mode_column="label_mode")
        recollect = True

    # Convert type
    if _has_faulty_image_type(df=df, img_type=img_type):
        typeconverter.convert_imgs_to_type(df, img_type, 
                                           path_column="image_path", suffix_column="image_suffix")
        typeconverter.convert_imgs_to_type(df, img_type, 
                                           path_column="label_path", suffix_column="label_suffix")
        recollect = True
        

    # Recollect metadata after conversions
    if recollect:
        df = _get_meta_data_with_labels(search_dir, supported_types)

    df.to_csv(metadata_path, index=False)
    logger.info(f"Metadata saved to '{metadata_path}'")


def _get_meta_data_with_labels(search_dir: str,types: List[str]) -> pd.DataFrame:
    """
    Walks through each task, collecting metadata from image/ and label/ pairs
    """
    all_rows = []
    root = Path(search_dir)

    for task_dir in root.iterdir():
        if not task_dir.is_dir():
            continue

        image_dir = task_dir / "image"
        label_dir = task_dir / "label"
        if not image_dir.exists() or not label_dir.exists():
            logger.warning(f"Missing image/ or label/ directory in {task_dir}")
            continue

        image_files = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in types])
        label_files = sorted([f for f in label_dir.iterdir() if f.suffix.lower() in types])

        if len(image_files) != len(label_files):
            logger.warning(f"Image/label count mismatch in {task_dir.name}")
            continue


        dataset_name = task_dir.name
        task_name = get_task_based_on_dataset(str(dataset_name))
        for img_path, lbl_path in zip(image_files, label_files):
            try:
                with Image.open(img_path) as img:
                    img_mode, img_size = img.mode, img.size
                with Image.open(lbl_path) as lbl:
                    lbl_mode, lbl_size = lbl.mode, lbl.size
            except Exception as e:
                logger.error(f"Error reading image/label: {e}")
                continue

            all_rows.append({
                "dataset": dataset_name,
                "task": task_name,
                "image_dir": str(image_dir),
                "image_name": img_path.stem,
                "image_suffix": img_path.suffix,
                "image_path" : str(img_path),
                "image_mode": img_mode,
                "image_width": img_size[0],
                "image_height": img_size[1],
                "label_dir": str(label_dir),
                "label_name": lbl_path.stem,
                "label_suffix": lbl_path.suffix,
                "label_path": str(lbl_path),
                "label_mode": lbl_mode,
                "label_width": lbl_size[0],
                "label_height": lbl_size[1],
            })

    return pd.DataFrame(all_rows)


def _has_faulty_image_type(df: pd.DataFrame, img_type: str = ".png") -> bool:
    """
    Returns True if any image or label has suffix != desired_type
    """
    return (
        not df["image_suffix"].eq(img_type).all() or
        not df["label_suffix"].eq(img_type).all()
    )


def _has_faulty_image_mode(df: pd.DataFrame, img_mode: str) -> bool:
    """
    Returns True if any image or label is not in desired_mode
    """
    return (
        not df["image_mode"].eq(img_mode).all() or
        not df["label_mode"].eq(img_mode).all()
    )
def get_task_based_on_dataset(dataset_name: str) -> str:
    """Function that maps a dataset to its task

    :param dataset_name: str -- Name of the dataset, e.g. "KERMANY"
    :return: task: str -- Task of the dataset, e.g. "classification"
    """
    task_mapping = {
        "mvtec_train": "inpainting",
        "ADE_20k": "segmentation",
        "rain13k": "deraining",
        "raise_deblur":"denoising",
        "raise_sr":"superesolution"
    }

    return task_mapping.get(dataset_name, "UNKNOWN TASK")

def get_metadata_df(path: str) -> pd.DataFrame:
    """Simply loads the metadata df, or dataframe given the input path.

    :param path: str -- Path to the metadata df
    :return: pd.DataFrame
    """
    assert Path(path).exists(), f"{path} not found!"

    df = pd.read_csv(path)
    logger.info(f"Loaded .csv file of shape {df.shape} at location '{path}'.")
    return df