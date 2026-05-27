"""
Module that handles the visualizations of the data using the metadata DataFrame.
"""

import logging

from PIL import Image

import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_segmentation(
    df: pd.DataFrame, idx: int, alpha: float = 0.33, fig_width: int = 18, fig_height: int = 5
) -> None:
    """
    Plots the instance identified by the index idx with its segmentation mask.

    :param df: pd.DataFrame -- Metadata df containing all the necessary information
    :param idx: int -- Index of the sample to visualize
    :param alpha: float -- The alpha of the segmentation mask in the overlayed plot
    :param fig_width: int -- Width of the figure
    :param fig_height: int -- Height of the figure
    :return: None
    """
    # Make sure only rows with segmentation masks are selected
    df = _filter_only_segmentation_mask(df)

    # Get paths to the image and segmentation mask
    img: str = df["path"].loc[idx]
    img_seg: str = df["segmentation_mask"].loc[idx]
    img_name: str = df["filename"].loc[idx]
    img_label: str = df["label"].loc[idx]
    img_dataset: str = df["dataset"].loc[idx]

    # PLOT SETTINGS
    fig, axes = plt.subplots(figsize=(fig_width, fig_height), nrows=1, ncols=3)
    # Side by Side
    axes[0].set_title(
        f"+++ Input +++\n" + f"Dataset: {img_dataset}, Label: {img_label}\n" + img_name, fontweight="bold"
    )
    axes[0].imshow(Image.open(img), cmap="gray")
    axes[0].axis("off")
    axes[1].set_title(
        "+++ Segmentation Mask +++\n" + f"Dataset: {img_dataset}, Label: {img_label}\n" + img_name, fontweight="bold"
    )
    axes[1].imshow(Image.open(img_seg))
    axes[1].axis("off")
    # Overlay image and mask
    axes[2].set_title(
        f"+++ Overlay (alpha = {alpha}) +++\n" + f"Dataset: {img_dataset}, Label: {img_label}\n" + img_name,
        fontweight="bold",
    )
    axes[2].imshow(Image.open(img), cmap="gray")
    axes[2].imshow(Image.open(img_seg), alpha=alpha)
    axes[2].axis("off")

    fig.tight_layout()
    fig.show()


def plot_random_classification_image_samples(
    df: pd.DataFrame, dataset: str, image_column: str = "path", label_column: str = "label", ncols: int = 5
) -> None:
    """Plots (UNIQUE_LABELS x NCOLS) random samples (conditioned on label) from the classification dataset
    specified in input param 'dataset'.

    :param df: pd.DataFrame -- metadata df containing all the information about the datasets / images
    :param dataset: str -- Name of the dataset to visualize the samples and labels
    :param image_column: str -- Name of the column, which contains the path to the image
    :param label_column: str -- Name of the column, which contains the label of the image
    :param ncols: int -- Number of samples to visualize of each class (default: 5)
    :return: None
    """
    assert dataset in df["dataset"].unique(), f"Unknown Dataset! Valid values are {df['dataset'].unique()}"

    # Filter for dataset
    df = df[df["dataset"] == dataset]

    # Define size of subplots, nrows = number of unique classes, ncols = Number of samples to display per class
    unique_labels = df[label_column].unique()
    NROWS = len(unique_labels)

    fig, axes = plt.subplots(figsize=(4 * ncols, 2 * NROWS), nrows=NROWS, ncols=ncols)
    fig.suptitle(f"{dataset} Dataset (n={len(df)})", fontweight="bold", size=20)
    fig.subplots_adjust(hspace=0.5)

    # Iterate through the rows (each unique value has its own row)
    for i, label in enumerate(unique_labels):
        # Filter for the label and extract ncols samples
        df_temp = df[df[label_column] == label]
        samples = df_temp.sample(ncols)

        # Iterate through the columns (number of samples to display per row / label)
        for j in range(ncols):
            ax = axes[i, j]
            # Get information of the sample
            img = Image.open(samples[image_column].values[j])
            width = samples["width"].values[j]
            height = samples["height"].values[j]
            filename = samples["filename"].values[j]
            # Plot image and adjust settings
            ax.imshow(img, cmap="gray")
            ax.set_title(f"{filename}\nLabel: {label}, Size: ({width}, {height})")
            ax.axis("off")

    fig.tight_layout()
    plt.show()


def _filter_only_segmentation_mask(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the dataframe for only instances / rows, where a segmentation mask is available.
    :param df: pd.DataFrame -- Metadata df
    :return: pd.DataFrame -- Filtered dataframe
    """
    return df[~df["segmentation_mask"].isnull()]
