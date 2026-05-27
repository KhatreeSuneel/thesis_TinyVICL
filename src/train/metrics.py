"""Module handling the metrics computation"""

import torch
import logging
import numpy as np

from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score

from src.train.postprocessing import post_process_to_indices
from src.utils.utils import get_tasks_for_rmse
from src.utils.tasks_utils import is_segmentation_task, is_edge_detection_task

from typing import List

logger = logging.getLogger(__name__)


def compute_metrics(y_true: torch.tensor, y_pred: torch.tensor, class_name: str, device) -> dict:
   

    """Function to compute metrics based on the given task

    For segmentation like tasks: Compute IoU (Jaccard) 
    For edge detection tasks: Compute F1-Score 
    For Rest: Compute RMSE and MAE 
    :param y_true: torch.tensor -- Ground Truth Image
    :param y_pred: torch.tensor -- Prediction Image
    :param class_name: str -- Sampled class name to differentiate which metric to use
    :param device -- Cuda device
    :return: metrics: dict
    """
    rmse_task_list = get_tasks_for_rmse()
    # Compute RMSE and MAE for non-segmentation like tasks
    if class_name in rmse_task_list:
        rmse, mae = _compute_rmse_and_mae(y_true, y_pred, device)
        return {"rmse": rmse, "mae": mae}
    
    if is_edge_detection_task(class_name):
        return _compute_edge_detection_metrics(y_true, y_pred, device)


    # Map image to the closest pixel and then obtain list of class indices for metric calculation
    idx_y_list, idx_y_pred_list = post_process_to_indices(y_true, y_pred)

    # Compute IoU for Segmentation  Tasks
    if is_segmentation_task(class_name):
        iou_micro = (
            torch.tensor([jaccard_score(y_true=idx_y_list, y_pred=idx_y_pred_list, average="micro")]).float().to(device)
        )
        iou_macro = (
            torch.tensor([jaccard_score(y_true=idx_y_list, y_pred=idx_y_pred_list, average="macro")]).float().to(device)
        )
        iou_weighted = (
            torch.tensor([jaccard_score(y_true=idx_y_list, y_pred=idx_y_pred_list, average="weighted")])
            .float()
            .to(device)
        )
        return {"iou_micro": iou_micro, "iou_macro": iou_macro, "iou_weighted": iou_weighted}

    


def _compute_rmse_and_mae(y_true, y_pred, device):
    """Metrics for non-segmentation like Tasks, e.g. Inpainting, Gaussian Deblurring, Image Invert, ...

    :param y_true: torch.tensor -- Ground Truth of shape (3, 192, 192)
    :param y_pred: torch.tensor -- Prediction of shape (3, 192, 192)
    :param device -- Cuda device
    """
    # Compute Mean Absolute Error (MAE)
    mae = torch.mean(torch.abs(y_true - y_pred))

    # Compute Root Mean Squared Error (RMSE)
    mse = torch.mean((y_true - y_pred) ** 2)
    rmse = torch.sqrt(mse)

    return torch.tensor([rmse]).float().to(device), torch.tensor([mae]).float().to(device)

def _compute_edge_detection_metrics(y_true, y_pred, device):
    """Metrics for Edge Detection Tasks

    Computes Precision, Recall, and F1-Score between predicted and ground truth
    edge maps. Assumes y_true and y_pred are binary or continuous maps in [0, 1].

    :param y_true: torch.tensor -- Ground Truth Edge Map of shape (1, H, W)
    :param y_pred: torch.tensor -- Predicted Edge Map of shape (1, H, W)
    :param device -- Cuda device
    """
    # If input is HxWx3, take first channel
    if y_true.ndim == 3 and y_true.shape[2] == 3:
        y_true = y_true[..., 0]
    if y_pred.ndim == 3 and y_pred.shape[2] == 3:
        y_pred = y_pred[..., 0]

    # Convert tensors to numpy and binarize
    y_true_bin = (y_true.detach().cpu().numpy() > 0.5).astype("uint8")
    y_pred_bin = (y_pred.detach().cpu().numpy() > 0.5).astype("uint8")

    # Flatten for sklearn metrics
    y_true_flat = y_true_bin.flatten()
    y_pred_flat = y_pred_bin.flatten()

    # Compute metrics
    precision = torch.tensor([precision_score(y_true_flat, y_pred_flat, zero_division=0)]).float().to(device)
    recall = torch.tensor([recall_score(y_true_flat, y_pred_flat, zero_division=0)]).float().to(device)
    f1 = torch.tensor([f1_score(y_true_flat, y_pred_flat, zero_division=0)]).float().to(device)

    return {"precision": precision, "recall": recall, "f1": f1}

