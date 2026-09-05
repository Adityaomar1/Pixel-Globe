import os
import numpy as np
import matplotlib.pyplot as plt
import torch


def denormalize_image(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Convert normalized PyTorch tensor (3, H, W) to (H, W, 3) uint8 image."""
    img = tensor.cpu().detach().numpy()
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    img = (img * np.array(std) + np.array(mean)) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def create_error_map(gt, pred):
    """
    Creates an RGB error visual map:
    - Green (0, 255, 0): True Positive (Hit)
    - Red (255, 0, 0): False Positive (False Alarm)
    - Blue (0, 0, 255): False Negative (Missed Detection)
    - Gray/Black (30, 30, 30): True Negative
    """
    h, w = gt.shape
    error_map = np.full((h, w, 3), 30, dtype=np.uint8)

    tp = (gt == 1) & (pred == 1)
    fp = (gt == 0) & (pred == 1)
    fn = (gt == 1) & (pred == 0)

    error_map[tp] = [0, 255, 0]    # Green: TP
    error_map[fp] = [255, 50, 50]  # Red: FP
    error_map[fn] = [50, 100, 255] # Blue: FN
    return error_map


def save_comparison_figure(t1, t2, gt, pred, save_path, sample_name=""):
    """
    Saves a 5-panel comprehensive visual evaluation figure:
    [T1 Image] | [T2 Image] | [Ground Truth] | [Prediction] | [Error Analysis Map]
    """
    if isinstance(t1, torch.Tensor):
        t1_np = denormalize_image(t1)
    else:
        t1_np = t1

    if isinstance(t2, torch.Tensor):
        t2_np = denormalize_image(t2)
    else:
        t2_np = t2

    if isinstance(gt, torch.Tensor):
        gt_np = (gt.cpu().detach().numpy().squeeze() > 0.5).astype(np.uint8)
    else:
        gt_np = (gt > 0.5).astype(np.uint8)

    if isinstance(pred, torch.Tensor):
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        pred_np = (pred.cpu().detach().numpy().squeeze() > 0.5).astype(np.uint8)
    else:
        pred_np = (pred > 0.5).astype(np.uint8)

    error_map = create_error_map(gt_np, pred_np)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), dpi=150)
    fig.suptitle(f"Change Detection Sample: {sample_name}", fontsize=14, fontweight='bold')

    axes[0].imshow(t1_np)
    axes[0].set_title("Time 1 ($T_1$) Image", fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(t2_np)
    axes[1].set_title("Time 2 ($T_2$) Image", fontsize=11)
    axes[1].axis('off')

    axes[2].imshow(gt_np, cmap='gray')
    axes[2].set_title("Ground Truth Mask", fontsize=11)
    axes[2].axis('off')

    axes[3].imshow(pred_np, cmap='gray')
    axes[3].set_title("TinyCD Prediction", fontsize=11)
    axes[3].axis('off')

    axes[4].imshow(error_map)
    axes[4].set_title("Error Map\n(TP: Green, FP: Red, FN: Blue)", fontsize=10)
    axes[4].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
