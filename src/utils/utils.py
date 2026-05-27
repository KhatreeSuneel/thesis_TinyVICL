""" Module containing utility / helper functions """

import argparse
import time
import yaml
import logging

from PIL import Image
from PIL.Image import Resampling
from torchvision.transforms.functional import pil_to_tensor

from typing import Dict, List, Union

logger = logging.getLogger(__name__)


def parse_args() -> Dict:
    """
    Defines and parses CLI arguments as a dictionary.

    :return: args_dict: dict
    """
    logger.info("Parsing CLI arguments as dict ...")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="Path to the config file",
        type=str,
        default="configs/config.yaml",
    )
    parser.add_argument(
        "--fit-neuralizer",
        help="Will fit Neuralizer on all trainings tasks",
        dest="fit_neuralizer",
        action="store_true",
    )
    parser.add_argument(
        "--fit-single-task-models",
        help="Will fit Single-Task Neuralizer Models: For each task, one model",
        dest="fit_single_task_models",
        action="store_true",
    )
    parser.add_argument(
    "--loss_name",
    type=str,
    default="mse",
    choices=["mse", "softdice", "mse+dssim", "task_specific","mae","mae+dssim"],
    help="Loss function to use (task_specific = per-task batch loss).",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.5,
        help="Lambda coefficient for MSE+DSSIM.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=11,
        help="Window size for DSSIM loss.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Ckpt for finetuning the model"
    )
    
    
    args = parser.parse_args()
    args_dict = vars(args)

    logger.info(f"Successfully parsed {len(args_dict.keys())} CLI argument(s).")

    return args_dict


def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """
    Loads a YAML configuration file as a dict given the path.

    :param config_path: str -- Path to the configuration file (default: "configs/config.yaml")
    :return: dict -- Configuration dictionary
    """
    try:
        with open(config_path, "r") as yamlfile:
            logger.info(f"Loading yaml file at '{config_path}' ...")
            return yaml.load(yamlfile, yaml.FullLoader)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {config_path} not found!")
    except PermissionError:
        raise PermissionError(f"Insufficient permission to read {config_path}!")
    except IsADirectoryError:
        raise IsADirectoryError(f"{config_path} is a directory!")


def setup_logging(loglevel=logging.INFO) -> None:
    """Handles the logger setup / configuration

    :param loglevel: Level of logging, e.g. {logging.DEBUG, logging.INFO}
    :return: None
    """
    logging.basicConfig(
        level=loglevel,
        format="%(asctime)s [%(name)s:%(lineno)s] [%(levelname)s] >>> %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_list_of_subclasses_of_class(parent_cls) -> List[str]:
    """Utility function to get the subclasses name of a given class

    :param parent_cls: Parent class to get subclasses name from
    :return: List[str] -- List of subclasses names
    """
    return [cls.__name__ for cls in parent_cls.__subclasses__()]

def get_tasks_for_rmse(config_path: str = "configs/config.yaml") -> List[str]:
    config = load_config(config_path)
    return config.get("non_segmentation_tasks", [])

def get_edge_detection_tasks(config_path: str = "configs/config.yaml") -> List[str]:
    config = load_config(config_path)
    return config.get("edge_detection_tasks", [])

def get_tasks_for_segmentation (config_path: str = "configs/config.yaml") -> List[str]:
    config = load_config(config_path)
    return config.get("segmentation_tasks", [])

def get_date_and_time_as_string() -> str:
    """Simple utility function to return the date and time as a string valid for filenames

    Example: 20241006-150603
    :return: str -- Current date and time as str
    """
    return time.strftime("%Y%m%d-%H%M%S")
def precompute_class_presence(task):
    """
    Optionally precompute unique label indices for each sample in a segmentation task.
    Triggered only if 'precompute_class_presence' flag is True in config.yaml.
    Resizing of labels is optional and follows the dataset config if present.
    """
    # --- Load config ---
    try:
        with open("configs/config.yaml", "r") as f:
            config = yaml.safe_load(f)
            enable_precompute = config.get("training", {}) \
                                      .get("dataset", {}) \
                                      .get("precompute_class_presence", False)
            resize_shape = config.get("training", {}) \
                                 .get("dataset", {}) \
                                 .get("resize_shape", None)
    except Exception as e:
        print(f"[Warning] Could not load config for precompute_class_presence: {e}")
        enable_precompute = False
        resize_shape = None

    if not enable_precompute:
        print(f"⚙️ Skipping class presence precomputation for {task.get_task_name()} (disabled in config).")
        return

    # --- Do precomputation ---
    df = task.get_data()
    class_presence = {}

    for i, row in df.iterrows():
        label_path = os.path.join(
            row["label_dir"],
            f"{row['label_name']}{row['label_suffix']}"
        )
        if not os.path.exists(label_path):
            print(f"⚠️ Label not found for index {i}: {label_path}")
            continue

        label_img = Image.open(label_path)
        if resize_shape:
            label_img = label_img.resize(resize_shape, Resampling.NEAREST)

        label_tensor = pil_to_tensor(label_img)
        unique_classes = torch.unique(label_tensor).tolist()
        class_presence[i] = set(unique_classes)

    task.class_presence = class_presence
    print(f"✅ Precomputed class presence for {task.get_task_name()} ({len(class_presence)} samples).")