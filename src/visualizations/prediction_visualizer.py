""" Module to visualize the predictions of our model """

import os
import torch
import logging
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Literal, Optional
from torchvision.transforms.v2.functional import to_pil_image
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from src.data.segmentation_mappings import SegmentationMappings
from src.train.metrics import compute_metrics
# from src.utils.tasks_utils import is_segmentation_or_segmentation_like_class


logger = logging.getLogger(__name__)


def plot_prediction_batch(
    x: torch.tensor,
    y: torch.tensor,
    y_pred: torch.tensor,
    ctx_out: torch.Tensor,
    class_names: List[str],
    print_metrics: bool = False,
    savefig: bool = False,
    save_path: str = "assets/predictions/",
    savefig_type: Literal["png", "svg"] = "png",
    suffix_fig: Optional[str] = None,
) -> None:
    """Plots the Input x, the prediction y_pred and the ground truth y in a grid like figure

    :param x: torch.tensor -- Input
    :param y: torch.tensor -- Ground Truth
    :param y_pred: torch.tensor -- Prediction
    :param ctx_out: torch.Tensor -- Ground Truth of Context
    :param class_names: List[str] -- List of Class Names of each Task
    :param print_metrics: bool -- Whether to print metrics or not (default: False)
    :param savefig: bool -- Whether to save the figure or not (default: False)
    :param save_path: str -- Path to save to
    :param savefig_type: Literal[".png", ".svg"] -- Type to save img as
    :param suffix_fig Optional[str] -- Suffix to add to the saved image name, e.g. "x.png" -> "x_Duke.png"
    :return: None
    """
    n_preds: int = y_pred.shape[0]
    # has_only_segmentation_like_tasks: bool = all([is_segmentation_or_segmentation_like_class(x) for x in class_names])

    # Check whether all tasks of this batch is segmentation like; If so, provide post processed row
    # if has_only_segmentation_like_tasks:
    #     nrows = 4
    # else:
    #     nrows = 3
    
    nrows = 3

    fig, axes = plt.subplots(
        nrows=nrows, ncols=n_preds, figsize=(2 * n_preds, int(nrows * 2.5)), sharex=True, sharey=True
    )
    fig.suptitle("Comparison Input, Prediction and Ground Truth".upper(), fontweight="bold", y=1.01, fontsize=16)
    for i in range(n_preds):
        img_x = to_pil_image(x[i, :, :, :])
        img_true = to_pil_image(y[i, :, :, :])
        img_pred = to_pil_image(y_pred[i, :, :, :])
        axes[0][i].set_title(f"{class_names[i]}\nInput $X_{i}$", fontweight="bold")
        axes[0][i].imshow(img_x)
        axes[1][i].set_title(rf"Prediction $pred_{i}$")
        axes[1][i].imshow(img_pred)
        # if has_only_segmentation_like_tasks:
        #     _, img_pred_postprocessed = map_image_to_closest_class_idx(y[i], y_pred[i], ctx_out[i], device="cpu")
        #     img_pred_postprocessed, _ = post_process_prediction(img_pred_postprocessed, ctx_out[i])
        #     axes[2][i].set_title(rf"Postprocessed $pred_{i}^*$")
        #     axes[2][i].imshow(to_pil_image(img_pred_postprocessed))
        #     axes[3][i].set_title(f"Ground Truth $Y_{i}$")
        #     axes[3][i].imshow(img_true)
        # else:
        axes[2][i].set_title(f"Ground Truth $Y_{i}$")
        axes[2][i].imshow(img_true)

        # Compute and print metrics
        if print_metrics:
            metric_dict = compute_metrics(y[i], y_pred[i], ctx_out[i], class_names[i], device="cpu")
            print(f"metric(pred_{i}^*, Y_{i}):\t {metric_dict}")

    fig.tight_layout()

    if savefig:
        save_path = Path(save_path)
        save_path = save_path.joinpath(f"prediction_batch{suffix_fig}.{savefig_type}")
        if not os.path.exists(save_path.parent):
            logger.info(f"'{save_path.parent}' non existent. Creating directory ...")
            os.makedirs(save_path.parent)
        logger.info(f"Saving figure to '{save_path}' ...")
        fig.savefig(save_path, bbox_inches="tight")


def plot_single_prediction(x: torch.tensor, y_pred: torch.tensor, alpha: float = 0.67) -> None:
    """Plots the ground truth x, the prediction y_pred and its overlay image with x being the img 'behind'

    :param x: torch.tensor -- Input
    :param y_pred: torch.tensor -- Prediction
    :param alpha: float -- Alpha level / overlay strength of the prediction
    :return: None
    """
    assert x.ndim == 3
    assert y_pred.ndim == 3

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 3), sharey=True)
    axes[0].imshow(to_pil_image(x))
    axes[0].set_title("Input $X$")
    axes[1].imshow(to_pil_image(y_pred))
    axes[1].set_title("Prediction")
    axes[2].imshow(to_pil_image(x))
    axes[2].imshow(to_pil_image(y_pred), alpha=alpha)
    axes[2].set_title("Overlay $X$ and Prediction")
    fig.tight_layout()


def plot_input_output_and_context_set(
    x: torch.Tensor,
    y_pred: torch.Tensor,
    ctx_in: torch.Tensor,
    ctx_out: torch.Tensor,
    task_name: str,
    class_name: str,
    border_width: int = 6,
):
    """Function to plot both the input and the output (postprocessed) as well as the context set

    :param x: torch.Tensor -- Input of the model
    :param y_pred: torch.Tensor -- Raw output of the model (unprocessed)
    :param ctx_in: torch.Tensor -- Context Set Input
    :param ctx_out: torch.Tensor -- Context Set Output
    :param task_name: str -- Name of the Task
    :param class_name: str -- Class Name of the Task
    :param border_width: int -- Line width of the border
    :return: None
    """
    n_context_size: int = ctx_in.shape[0]

    # If our task is a segmentation like task, postprocess the prediction to map colors to closest valid color
    # if is_segmentation_or_segmentation_like_class(class_name):
    #     _, y_pred_mapped = map_image_to_closest_class_idx(
    #         torch.rand((3, 192, 192)), y_pred, ctx_out, device=y_pred.device
    #     )
    #     y_pred, _ = post_process_prediction(y_pred_mapped, ctx_out)

    fig = plt.figure(layout="constrained", figsize=(12, 4))
    gs = GridSpec(2, n_context_size + 1, figure=fig)

    fig.suptitle(task_name, size=18, x=0.03, horizontalalignment="left", verticalalignment="top")

    ax1 = fig.add_subplot(gs[0, 0])  # Input
    ax1.imshow(to_pil_image(x))
    # ax1.set_title("Input $X$")
    ax1.patch.set_edgecolor("orange")
    ax1.patch.set_linewidth(border_width)

    ax2 = fig.add_subplot(gs[1, 0])  # Pred
    ax2.imshow(to_pil_image(y_pred))
    # ax2.set_title("Prediction $y^*_{pred}$")
    ax2.patch.set_edgecolor("violet")
    ax2.patch.set_linewidth(border_width)

    # Context Set
    for j in range(1, n_context_size + 1):
        # Context In
        ax_ctx_in = fig.add_subplot(gs[0, j], sharey=ax1)
        plt.setp(ax_ctx_in.get_yticklabels(), visible=False)
        ax_ctx_in.imshow(to_pil_image(ctx_in[j - 1]))
        # ax_ctx_in.set_title(f"Context Input {j-1}")
        ax_ctx_in.patch.set_edgecolor("grey")
        ax_ctx_in.patch.set_linewidth(border_width)

        # Context Out
        ax_ctx_out = fig.add_subplot(gs[1, j], sharey=ax2)
        plt.setp(ax_ctx_out.get_yticklabels(), visible=False)
        ax_ctx_out.imshow(to_pil_image(ctx_out[j - 1]))
        # ax_ctx_out.set_title(f"Context Output {j-1}")
        ax_ctx_out.patch.set_edgecolor("grey")
        ax_ctx_out.patch.set_linewidth(border_width)

    legend_elements = [
        Patch(facecolor="white", edgecolor="orange", label="Input", linewidth=2),
        Patch(facecolor="white", edgecolor="violet", label="Model Prediction", linewidth=2),
        Patch(facecolor="white", edgecolor="grey", label="Context Set", linewidth=2),
    ]
    fig.legend(
        handles=legend_elements, loc="upper right", handlelength=1.4, handleheight=1.4, ncol=len(legend_elements)
    )
    plt.show()
    return fig
