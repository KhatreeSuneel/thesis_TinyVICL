import torch
import torch.nn as nn
from typing import Dict
import torch.nn.functional as F


# -------------------------
# Base Loss Wrappers
# -------------------------

class MSELossWrapper(nn.Module):
    """Standard MSE Loss wrapper."""
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction="mean")

    def forward(self, y_true, y_pred):
        return self.loss(y_pred, y_true)

class L1LossWrapper(nn.Module):
    """Standard L1 Loss wrapper."""
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction="mean")

    def forward(self, y_true, y_pred):
        return self.loss(y_pred, y_true)
    
class SoftDiceLoss(nn.Module):
    """Soft Dice Loss for RGB segmentation outputs."""
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        # Flatten spatial dimensions but keep channels separate
        B, C, H, W = y_pred.shape
        y_pred_flat = y_pred.view(B, C, -1)
        y_true_flat = y_true.view(B, C, -1)

        intersection = (y_pred_flat * y_true_flat).sum(dim=2)  # sum over pixels
        union = (y_pred_flat ** 2).sum(dim=2) + (y_true_flat ** 2).sum(dim=2)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()  # average over batch and channels


class DSSIMLoss(nn.Module):
    """
    Structural dissimilarity (DSSIM) loss.
    Requires `pytorch-msssim` to be installed:
        pip install pytorch-msssim
    """
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size

    def forward(self, y_true, y_pred):
        from pytorch_msssim import ssim  
        return 1 - ssim(
            y_pred, y_true,
            data_range=1.0,
            size_average=True,
            win_size=self.window_size
        )


class MSEPlusDSSIM(nn.Module):
    """Combined MSE + λ * DSSIM."""
    def __init__(self, lam: float = 0.5, window_size: int = 11):
        super().__init__()
        self.mse = nn.MSELoss(reduction="mean")
        self.dssim = DSSIMLoss(window_size=window_size)
        self.lam = lam

    def forward(self, y_true, y_pred):
        return self.mse(y_pred, y_true) + self.lam * self.dssim(y_true, y_pred)
    
class MAEPlusDSSIM(nn.Module):
    """Combined MSE + λ * DSSIM."""
    def __init__(self, lam: float = 0.5, window_size: int = 11):
        super().__init__()
        self.mae = nn.L1Loss(reduction="mean")
        self.dssim = DSSIMLoss(window_size=window_size)
        self.lam = lam

    def forward(self, y_true, y_pred):
        return self.mae(y_pred, y_true) + self.lam * self.dssim(y_true, y_pred)

class CosineSoftDiceLoss(nn.Module):
    """
    Soft Dice Loss for RGB predictions vs. discrete color targets, using cosine similarity.
    - Computes per-image palette of unique target colors.
    - Converts predicted RGB to class-probabilities via cosine similarity + softmax.
    - Computes soft dice against binary maps for each color class.
    """
    def __init__(self, smooth: float = 1e-6, tau: float = 1.0, per_present_class: bool = True,print_every: int = 1500):
        super().__init__()
        self.smooth = smooth
        self.tau = tau
        self.per_present_class = per_present_class
        self.print_every = print_every
        self._step_counter = 0 

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_rgb:  (B, 3, H, W) predicted RGB values
            target_rgb:(B, H, W, 3) discrete GT colors
        Returns:
            scalar loss averaged over batch
        """
        B, _, H, W = y_pred.shape
        loss_sum = 0.0

        # flatten spatial dims
        pred_flat = y_pred.permute(0, 2, 3, 1).reshape(B, -1, 3)   # (B, HW, 3)
        tgt_flat  = y_true.permute(0, 2, 3, 1).reshape(B, -1, 3)   # (B, HW, 3)

        for b in range(B):
            # palette of unique colors for this image[]
            colors = torch.unique(tgt_flat[b], dim=0)                 # (K, 3)
            K = colors.shape[0]
            if self._step_counter==0 or self._step_counter%self.print_every==0:
                print(f"[Step {self._step_counter}] Image {b} unique colors: {K}")
                
            # predictions and palette
            p = F.normalize(pred_flat[b], dim=-1)                     # (HW, 3) 
            c = F.normalize(colors, dim=-1)                           # (K, 3)                          

            sims = p @ c.T                                            # (HW, K)
            probs = F.softmax(sims / self.tau, dim=-1)                # (HW, K)

            # binary GT per class
            y = (tgt_flat[b].unsqueeze(1) == colors.unsqueeze(0)).all(-1).float()  # (HW, K)

            # Dice numerator & denominator
            numer = 2 * (probs * y).sum(0)                            # (K,)
            denom = probs.sum(0) + y.sum(0) + self.smooth             # (K,)

            if self.per_present_class:
                present = (y.sum(0) > 0).float()                      # (K,)
                dice_k = torch.where(present > 0, numer / denom, torch.zeros_like(numer))
                dice = dice_k.sum() / (present.sum() + self.smooth)
            else:
                dice = (numer / denom).mean()

            loss_sum += (1 - dice)
        self._step_counter+=1

        return loss_sum / B
    
    
class EuclideanSoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, tau=1.0, per_present_class=True, print_every=1500):
        super().__init__()
        self.smooth = smooth
        self.tau = tau
        self.per_present_class = per_present_class
        self.print_every = print_every
        self._step_counter = 0

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_true: (B, 3, H, W) ground truth RGB mask
            y_pred: (B, 3, H, W) predicted RGB
        Returns:
            scalar loss
        """
        B, _, H, W = y_pred.shape
        loss_sum = 0.0

        y_pred_flat = y_pred.permute(0, 2, 3, 1).reshape(B, -1, 3)
        y_true_flat = y_true.permute(0, 2, 3, 1).reshape(B, -1, 3)

        for b in range(B):
            colors = torch.unique(y_true_flat[b], dim=0)
            K = colors.shape[0]
            if self._step_counter % self.print_every == 0:
                print(f"[Step {self._step_counter}] Euclidean Loss | Image {b}: {K} unique colors")

            # Euclidean distance → similarity
            dists = torch.cdist(y_pred_flat[b].unsqueeze(0), colors.unsqueeze(0)).squeeze(0)  # (HW, K)
            sims = -dists                                           # higher = closer
            sims = sims - sims.max(dim=-1, keepdim=True)[0]         # stabilize
            probs = F.softmax(sims / self.tau, dim=-1)

            # one-hot GT
            y = (y_true_flat[b].unsqueeze(1) == colors.unsqueeze(0)).all(-1).float()

            numer = 2 * (probs * y).sum(0)
            denom = probs.sum(0) + y.sum(0) + self.smooth

            if self.per_present_class:
                present = y.sum(0) > 0
                dice = (numer[present] / denom[present]).mean() if present.any() else torch.tensor(0.0, device=y.device)
            else:
                dice = (numer / denom).mean()

            loss_sum += (1 - dice)

        self._step_counter += 1
        return loss_sum / B


class CosineSoftDicePlusSoftDice(nn.Module):
    """
    Combination of CosineSoftDiceLoss and SoftDiceLoss.
    """
    def __init__(self, alpha: float = 0.5, **kwargs):
        """
        Args:
            alpha: weight for CosineSoftDiceLoss. (1-alpha) is used for SoftDiceLoss.
        """
        super().__init__()
        self.alpha = alpha
        self.cosine_softdice = CosineSoftDiceLoss(**kwargs)
        self.softdice = SoftDiceLoss()

    def forward(self, y_true, y_pred):
        loss_cosine = self.cosine_softdice(y_true, y_pred)
        loss_soft = self.softdice(y_true, y_pred)
        return self.alpha * loss_cosine + (1 - self.alpha) * loss_soft



# -------------------------
# Task-Specific Wrapper
# -------------------------

TASK_LOSSES: Dict[str, nn.Module] = {
    "mse": MSELossWrapper(),
    "softdiceloss": SoftDiceLoss(),
    "cosinesoftdice": CosineSoftDiceLoss(),
    "euclideansoftdice": EuclideanSoftDiceLoss(),
    "mse+dssim": MSEPlusDSSIM(lam=0.5),
    "combinedsoftdice" : CosineSoftDicePlusSoftDice(alpha=0.5)
}


class TaskSpecificLossModule(nn.Module):
    """
    Chooses a loss dynamically per batch, based on the lossfun string
    provided in the dataset batch.
    """

    def __init__(self, ):
        super().__init__()
        # Use default TASK_LOSSES if none provided
        self.task_losses = TASK_LOSSES 
    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor, lossfun):
        """
        Computes per-sample loss based on provided lossfun list/tuple.
        Args:
            y_true: Tensor of ground truth images, shape [B, C, H, W]
            y_pred: Tensor of predictions, shape [B, C, H, W]
            lossfun: List or tuple of loss names, one per batch element
        Returns:
            Tensor of averaged loss over the batch
        """
        if not isinstance(lossfun, (list, tuple)):
            raise ValueError(
                f"Expected lossfun to be list or tuple for task-specific loss, got {type(lossfun)}"
            )

        losses = []
        for i in range(len(y_true)):
            lf = str(lossfun[i]).lower()
            if lf not in self.task_losses:
                raise ValueError(f"Unknown task loss: {lf}")
            losses.append(self.task_losses[lf](y_true[[i]], y_pred[[i]]).view(1))

        return torch.cat(losses, dim=0).mean()


# -------------------------
# Loss Factory
# -------------------------

def get_loss_fn(loss_name: str, **kwargs) -> nn.Module:
    """
    Factory for dynamic loss selection.
    Args:
        loss_name: one of ["mse", "softdice", "mse+dssim", "task_specific"]
        kwargs: extra args (e.g. lam for mse+dssim)
    """
    loss_name = loss_name.lower()

    if loss_name == "mse":
        return MSELossWrapper()
    elif loss_name == "mae":
        return L1LossWrapper()
    elif loss_name =="mae+dssim":
        lam = kwargs.get("lam", 0.5)
        window_size = kwargs.get("window_size", 11)
        return MAEPlusDSSIM(lam=lam, window_size=window_size)
    elif loss_name == "mse+dssim":
        lam = kwargs.get("lam", 0.5)
        window_size = kwargs.get("window_size", 11)
        return MSEPlusDSSIM(lam=lam, window_size=window_size)
    elif loss_name == "task_specific":
        return TaskSpecificLossModule()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
