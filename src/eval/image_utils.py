import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from lpips import LPIPS
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import precision_score, recall_score, f1_score

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize LPIPS model
lpips_model = LPIPS(net='alex').to(device)

transform = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.ToTensor()
])

def tensor_to_grayscale(tensor):
    """
    Convert tensor (B, C, H, W) or (C, H, W) to grayscale.
    Uses simple average over channels.
    """
    if tensor.ndim == 4:
        # (B, C, H, W)
        return tensor.mean(dim=1, keepdim=True)
    elif tensor.ndim == 3:
        # (C, H, W)
        return tensor.mean(dim=0, keepdim=True)
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")


def safe_binarize(pred, threshold):
    """
    Binarize after grayscale conversion and return (H, W) uint8 array.
    """
    # Convert to grayscale
    if isinstance(pred, torch.Tensor):
        pred_gray = tensor_to_grayscale(pred)
    else:
        raise ValueError("pred should be a torch.Tensor")

    bin_array = (pred_gray > threshold).cpu().numpy().astype(np.uint8).squeeze()

    assert bin_array.ndim == 2, f"Unexpected shape after binarizing: {bin_array.shape}"
    return bin_array



def compute_metrics(pred, target, task="colorization", threshold=0.5,save=None,id=None):
    assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"
    
    pred_np = pred.detach().cpu().numpy().squeeze(0)
    target_np = target.detach().cpu().numpy().squeeze(0)
    
    pred_img = pred_np.transpose(1, 2, 0) if pred_np.ndim == 3 else pred_np
    target_img = target_np.transpose(1, 2, 0) if target_np.ndim == 3 else target_np

    metrics = {}

    # Common: MSE, PSNR, SSIM, LPIPS
    if task in ["colorization", "deraining"]:
        mse = F.mse_loss(pred, target).item()
        psnr = peak_signal_noise_ratio(target_img, pred_img, data_range=1.0)
        ssim = structural_similarity(target_img, pred_img, data_range=1.0, channel_axis=-1)
        lpips_val = lpips_model((pred * 2 - 1).to(device), (target * 2 - 1).to(device)).item()
        
        metrics.update({
            "MSE": mse,
            "PSNR": psnr,
            "SSIM": ssim,
            "LPIPS": lpips_val
        })

    # Depth estimation: RMSE
    elif task == "depth_estimation":
        rmse = torch.sqrt(F.mse_loss(pred, target)).item()
        metrics["RMSE"] = rmse

    # Edge detection: binarize and compute precision, recall, F1
    elif task == "edge_detection":
        # pred_bin = (pred > threshold).cpu().numpy().astype(np.uint8).squeeze()
        # target_bin = (target > threshold).cpu().numpy().astype(np.uint8).squeeze()
        # # print(f'pred_shape:{pred_bin.shape},target_shape:{target_bin.shape}')
        
        # pred_flat = pred_bin.flatten()
        # target_flat = target_bin.flatten()
        # Binarize after grayscale conversion
        pred_bin = safe_binarize(pred, threshold)
        target_bin = safe_binarize(target, threshold)

        # Metrics (flattened arrays)
        pred_flat = pred_bin.flatten()
        target_flat = target_bin.flatten()

        precision = precision_score(target_flat, pred_flat, zero_division=0)
        recall = recall_score(target_flat, pred_flat, zero_division=0)
        f1 = f1_score(target_flat, pred_flat, zero_division=0)
        psnr = peak_signal_noise_ratio(target_img, pred_img, data_range=1.0)
        ssim = structural_similarity(target_img, pred_img, data_range=1.0,channel_axis=-1)

        metrics.update({
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "PSNR": psnr,
            "SSIM": ssim
        })
        # === Save visualizations if save directory is provided ===
        if save is not None:
            os.makedirs(save, exist_ok=True)

            # Convert to uint8 images for saving
            pred_img_save = Image.fromarray((pred_bin * 255).astype(np.uint8))
            target_img_save = Image.fromarray((target_bin * 255).astype(np.uint8))

            pred_img_save.save(os.path.join(save, f"{id}_pred_bin.png"))
            target_img_save.save(os.path.join(save, f"{id}_target_bin.png"))

    else:
        raise ValueError(f"Unknown task type: {task}")

    return metrics



