from PIL import Image
import numpy as np
import torch
import os
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (0x44, 0x01, 0x54)
YELLOW = (0xFD, 0xE7, 0x25)


def round_image(img, options=(WHITE, BLACK,PURPLE,YELLOW), outputs=None, t=(0, 0, 0)):
    # img.shape == [224, 224, 3], img.dtype == torch.int32
    img = torch.tensor(img)
    t = torch.tensor((t)).to(img)
    options = torch.tensor(options)
    opts = options.view(len(options), 1, 1, 3).permute(1, 2, 3, 0).to(img)
    nn = (((img + t).unsqueeze(-1) - opts) ** 2).float().mean(dim=2)
    nn_indices = torch.argmin(nn, dim=-1)
    if outputs is None:
        outputs = options
    res_img = torch.tensor(outputs)[nn_indices]
    return res_img

def calculate_metric(args, target, ours,purple):
    target = np.asarray(target)
    ours = np.asarray(ours)
    
    assert target.shape == ours.shape, f"Shape mismatch: {target.shape} vs {ours.shape}"
    if purple==0:
        fg_color = np.array(WHITE)
        bg_color = np.array(BLACK)
    else:
        fg_color = np.array(YELLOW)
        bg_color = np.array(PURPLE)

    accuracy = np.sum(np.float32((target == ours).all(axis=2))) / (ours.shape[0] * ours.shape[1])

    # Segmentation masks
    seg_orig = (target == fg_color).all(axis=2)
    seg_our = (ours == fg_color).all(axis=2)
    color_blind_seg_our = (ours != bg_color).any(axis=2)

    # IOU
    iou = np.sum(seg_orig & seg_our) / np.sum(seg_orig | seg_our)
    color_blind_iou = np.sum(seg_orig & color_blind_seg_our) / np.sum(seg_orig | color_blind_seg_our)

    return {'iou': iou,'color_blind_iou':color_blind_iou, 'accuracy': accuracy}

def threshold_metric(args, target, ours, threshold, save=None, basename="output"):
    assert target.shape == ours.shape, f"Shape mismatch: {target.shape} vs {ours.shape}"

    # Normalize to [0, 1]
    target = target / 255.0
    ours = ours / 255.0

    # Threshold to binary
    target = (target > threshold).astype(np.uint8)
    ours = (ours > threshold).astype(np.uint8)

    # Convert to 0/255 for RGB logic
    target *= 255
    ours *= 255

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    fg_color = np.array(WHITE)
    bg_color = np.array(BLACK)

    # Handle both grayscale and RGB images
    if target.ndim == 2:
        pixel_match = target == ours
        seg_orig = target == fg_color[0]
        seg_our = ours == fg_color[0]
        color_blind_seg_our = ours != bg_color[0]
    else:
        pixel_match = (target == ours).all(axis=2)
        seg_orig = (target == fg_color).all(axis=2)
        seg_our = (ours == fg_color).all(axis=2)
        color_blind_seg_our = (ours != bg_color).any(axis=2)

    accuracy = np.sum(pixel_match) / pixel_match.size
    iou = np.sum(seg_orig & seg_our) / np.sum(seg_orig | seg_our)
    color_blind_iou = np.sum(seg_orig & color_blind_seg_our) / np.sum(seg_orig | color_blind_seg_our)

    if save is not None:
        os.makedirs(save, exist_ok=True)
        Image.fromarray(ours.astype(np.uint8)).save(os.path.join(save, f"{basename}_pred_bin.png"))
        Image.fromarray(target.astype(np.uint8)).save(os.path.join(save, f"{basename}_target_bin.png"))
    else:
        raise ValueError("Output dir not specified")

    return {
        'iou': iou,
        'color_blind_iou': color_blind_iou,
        'accuracy': accuracy
    }