from tqdm import trange
import os
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import torchvision.transforms as T
from torchvision import transforms
import argparse
import src.utils.utils as utils
from src.dataset import image_dataset
from src.neuralizer.lightning_model import LightningModel
from src.eval.image_utils import compute_metrics


encode_transform = transforms.Compose(
    [
        transforms.Resize((256,256), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ]
)

def get_args():
    parser = argparse.ArgumentParser('Evaluate Neuralizer_General with Task-Specific Metrics')
    parser.add_argument('--csv_dir', required=True, help='Path to dataset csv file')
    parser.add_argument('--data_dir', required=True, help='Path to dataset')
    parser.add_argument('--label_dir', required=True, help='Path to target file')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--ckpt_path', required=True, help='Path to model')
    parser.add_argument('--num_shots', default=2, type=int)
    parser.add_argument('--task', required=True, choices=['colorization', 'deraining', 'depth_estimation', 'edge_detection'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threshold', default=0.5,type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--max_examples', type=int, default=1000)
    return parser

def evaluate(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("Loading models...")
   
    model= LightningModel.load_from_checkpoint(
        checkpoint_path=args.ckpt_path, learning_rate = 1e-4, batch_size=1,strict=False,
    )
    model = model.to(args.device)
    model.eval()

    dataset = image_dataset.Imagedataset(
        csv_path = args.csv_dir,
        image_dir = args.data_dir,
        label_dir = args.label_dir,
        image_transform = encode_transform,
        support = args.num_shots
        )
    
    metric_totals = {}
    count = 0

    for idx in trange(min(len(dataset), args.max_examples)):
        data = dataset[idx]

        # The dataset returns: {"input_images": ..., "query_image": ..., "query_label": ...}
        query_img = data["query_image"].unsqueeze(0).to(args.device)   # (1, C, H, W)
        query_lbl = data["query_label"].unsqueeze(0).to(args.device)   # (1, C, H, W)

        # Support images + labels (few-shot context)
        support_imgs = data["support_images"]
        support_lbls = data["support_labels"]
        
        # Stack along L dimension and add batch dimension
        support_imgs = torch.stack(support_imgs, dim=0).unsqueeze(0).to(args.device)  # (B=1, L, C, H, W)
        support_lbls = torch.stack(support_lbls, dim=0).unsqueeze(0).to(args.device)  # (B=1, L, C, H, W)


        # Neuralizer expects x, ctx_in, ctx_out
        x = query_img
        ctx_in = support_imgs
        ctx_out = support_lbls

        with torch.no_grad():
            y_pred = model.forward(x, ctx_in, ctx_out)   

        # Convert to PIL for saving + metrics
        pred_img_pil = T.ToPILImage()(y_pred.squeeze(0).cpu().clamp(0, 1))
        target_img_pil = T.ToPILImage()(query_lbl.squeeze(0).cpu())
        
        pred_tensor = T.ToTensor()(pred_img_pil).unsqueeze(0).to(args.device)
        target_tensor = T.ToTensor()(target_img_pil).unsqueeze(0).to(args.device)

        # Compute task-specific metrics
        metric_dict = compute_metrics(
            pred_tensor,
            target_tensor,
            task=args.task,
            threshold=args.threshold,
            save=args.output_dir,
            id=idx,
        )

        # Accumulate
        for k, v in metric_dict.items():
            metric_totals[k] = metric_totals.get(k, 0) + v
        count += 1

        # Save prediction + ground truth
        query_name = data["query_id"][0]
        out_name = os.path.join(args.output_dir, f"{idx}_{query_name}")
        pred_img_pil.save(f"{out_name}_pred.png")
        target_img_pil.save(f"{out_name}_gt.png")

    # Write average results
    output_file = os.path.join(args.output_dir, "metrics.txt")
    with open(output_file, "w") as f:
        print("\nFinal Task-Specific Metrics:")
        for k, total in metric_totals.items():
            avg = total / count
            print(f"{k}: {avg:.4f}")
            f.write(f"{k}: {avg:.4f}\n")

    print(f"\nEvaluation complete. Metrics saved to {output_file}")

if __name__ == '__main__':
    args = get_args().parse_args()
    evaluate(args)
