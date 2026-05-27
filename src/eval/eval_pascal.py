import os
import argparse
from pathlib import Path
from tqdm import trange
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T
from torchvision import transforms

from src.neuralizer.lightning_model import LightningModel
from src.eval.segmentation_utils import calculate_metric, round_image
from src.eval import pascal_dataset

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (0x44, 0x01, 0x54)
YELLOW = (0xFD, 0xE7, 0x25)
encode_transform = transforms.Compose(
    [
        transforms.Resize((256,256), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ]
)

def get_args():
    parser = argparse.ArgumentParser("Evaluate Neuralizer on Pascal few-shot segmentation")
    parser.add_argument("--base_dir", required=True, help="Path to Pascal VOC dataset")
    parser.add_argument("--output_dir", default="./output_eval")
    parser.add_argument("--ckpt_path", required=True, help="Path to Neuralizer checkpoint")
    parser.add_argument("--split", default=0, type=int)
    parser.add_argument("--num_shots", default=2, type=int)
    parser.add_argument('--purple', default=0, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max_examples", type=int, default=1000)
    return parser


def evaluate(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Loading Neuralizer model...")
    model = LightningModel.load_from_checkpoint(
        checkpoint_path=args.ckpt_path,
        learning_rate=1e-4,
        batch_size=1,
        strict=False,
    ).to(device)
    model.eval()

    print("Loading dataset...")
    dataset = pascal_dataset.DatasetPASCAL(
        datapath=args.base_dir,
        fold=args.split,
        image_transform=encode_transform,
        mask_transform=encode_transform,
        support=args.num_shots,
        purple=bool(args.purple)
    )

    metric_total = {"iou": 0, "color_blind_iou": 0, "accuracy": 0}

    for idx in trange(min(len(dataset), args.max_examples)):
        data = dataset[idx]
        
        support_imgs = data["support_images"]
        query_img = data["query_image"]

        support_masks = data["support_masks"]
        query_mask = data["query_mask"]
        query_name = data["query_id"]

        def ensure_three_channels(img):
            if img.shape[0] == 1:
                return img.repeat(3, 1, 1)
            return img[:3]

        # Prepare tensors
        support_imgs = torch.stack(
            [ensure_three_channels(x) for x in support_imgs], dim=0
        ).unsqueeze(0).to(device)        # (1, L, C, H, W)

        support_masks = torch.stack([ensure_three_channels(x) for x in support_masks], dim=0).unsqueeze(0).to(device)

        query_img = ensure_three_channels(query_img).unsqueeze(0).to(device)
        
        # Neuralizer expects x, ctx_in, ctx_out
        x = query_img
        ctx_in = support_imgs
        ctx_out = support_masks

        # -----------------------------
        # Neuralizer inference 
        # -----------------------------
        with torch.no_grad():
            y_pred = model.forward(x, ctx_in, ctx_out)  
        y_pred = y_pred.clamp(0, 1)

        # Convert outputs
        pred_pil = T.ToPILImage()(y_pred.squeeze(0).cpu())
        gt_pil = T.ToPILImage()(query_mask.cpu())

        pred_np = np.array(pred_pil)
        gt_np = np.array(gt_pil)
        if gt_np.ndim == 2:
            gt_np = np.stack([gt_np] * 3, axis=-1)
        
        pred_bin = round_image(pred_np, options=(WHITE, BLACK, PURPLE, YELLOW), outputs=None, t=(0, 0, 0))
        gt_bin = round_image(gt_np, options=(WHITE, BLACK, PURPLE, YELLOW), outputs=None, t=(0, 0, 0))

        # Save images
        out_name = os.path.join(args.output_dir, f"{idx}_{query_name}")
        pred_pil.save(f"{out_name}_generated.png")
        gt_pil.save(f"{out_name}_original.png")
        # Image.fromarray(np.uint8(pred_bin)).save(f"{out_name}_generated_rounded.png") 
        # Image.fromarray(np.uint8(gt_bin)).save(f"{out_name}_original_rounded.png")

        # Metrics
        metric = calculate_metric(args, gt_bin, pred_bin,args.purple)
        for k in metric_total:
            metric_total[k] += metric[k] / min(len(dataset), args.max_examples)

    print("\nFinal Metrics:")
    for k, v in metric_total.items():
        print(f"{k}: {v:.4f}")

    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
        for k, v in metric_total.items():
            f.write(f"{k}: {v:.4f}\n")


if __name__ == "__main__":
    args = get_args().parse_args()
    evaluate(args)
