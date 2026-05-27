import os
import argparse
from pathlib import Path
from tqdm import trange
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T
from torchvision import transforms

from src.eval.segmentation_utils import calculate_metric, round_image
from src.eval import pascal_dataset


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (0x44, 0x01, 0x54)
YELLOW = (0xFD, 0xE7, 0x25)

encode_transform = transforms.Compose(
    [
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ]
)


def get_args():
    parser = argparse.ArgumentParser("Evaluate Copy Baselines on Pascal few-shot segmentation")
    parser.add_argument("--base_dir", required=True, help="Path to Pascal VOC dataset")
    parser.add_argument("--output_dir", required=True, help="Path to save outputs")
    parser.add_argument("--split", default=0, type=int)
    parser.add_argument("--num_shots", default=2, type=int)
    parser.add_argument("--purple", default=0, type=int)
    parser.add_argument("--baseline", required=True, choices=["copy_context", "copy_query"])
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max_examples", type=int, default=1000)
    return parser


def ensure_three_channels(img):
    if img.shape[0] == 1:
        return img.repeat(3, 1, 1)
    return img[:3]


def evaluate(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    dataset = pascal_dataset.DatasetPASCAL(
        datapath=args.base_dir,
        fold=args.split,
        image_transform=encode_transform,
        mask_transform=encode_transform,
        support=args.num_shots,
        purple=bool(args.purple),
    )

    metric_total = {"iou": 0, "color_blind_iou": 0, "accuracy": 0}

    for idx in trange(min(len(dataset), args.max_examples)):
        data = dataset[idx]

        support_imgs = data["support_images"]
        support_masks = data["support_masks"]
        query_img = data["query_image"]
        query_mask = data["query_mask"]
        query_name = data["query_id"]

        # -----------------------------
        # Baseline prediction
        # -----------------------------
        if args.baseline == "copy_context":
            # Use first support mask as prediction
            pred_tensor = ensure_three_channels(support_masks[0])

        elif args.baseline == "copy_query":
            # Use query image itself as prediction
            pred_tensor = ensure_three_channels(query_img)

        # Convert to PIL
        pred_pil = T.ToPILImage()(pred_tensor)
        gt_pil = T.ToPILImage()(query_mask)

        pred_np = np.array(pred_pil)
        gt_np = np.array(gt_pil)

        if gt_np.ndim == 2:
            gt_np = np.stack([gt_np] * 3, axis=-1)

        # Round colors
        gt_bin = round_image(gt_np, options=(WHITE, BLACK), outputs=None, t=(0, 0, 0))
        if args.baseline == "copy_context":
            # Support mask needs rounding
            pred_bin = round_image(pred_np, options=(WHITE, BLACK), outputs=None, t=(0, 0, 0))

        elif args.baseline == "copy_query":
            # Do NOT round query image
            pred_bin = pred_np

        # -----------------------------
        # Save outputs
        # -----------------------------
        base_name = os.path.join(args.output_dir, f"{idx}_{query_name}")

        if args.baseline == "copy_context":
            Image.fromarray(np.uint8(pred_bin)).save(f"{base_name}_copy_context.png")

        elif args.baseline == "copy_query":
            Image.fromarray(np.uint8(pred_bin)).save(f"{base_name}_copy_query.png")

        Image.fromarray(np.uint8(gt_bin)).save(f"{base_name}_original.png")

        # -----------------------------
        # Metrics
        # -----------------------------
        metric = calculate_metric(args, gt_bin, pred_bin, args.purple)

        for k in metric_total:
            metric_total[k] += metric[k] / min(len(dataset), args.max_examples)

    # -----------------------------
    # Final metrics
    # -----------------------------
    print("\nFinal Metrics:")
    for k, v in metric_total.items():
        print(f"{k}: {v:.4f}")

    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
        for k, v in metric_total.items():
            f.write(f"{k}: {v:.4f}\n")


if __name__ == "__main__":
    args = get_args().parse_args()
    evaluate(args)