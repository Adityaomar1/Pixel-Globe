import os
import glob
import argparse
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from models import TinyCD


def parse_args():
    parser = argparse.ArgumentParser(description="Run Inference using Trained TinyCD Model (SIH PS 1518)")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint (e.g. checkpoints/best_model.pth)')
    
    # Input Modes (Single Pair OR Directory of Pairs)
    parser.add_argument('--t1', type=str, default=None, help='Path to single T1 image')
    parser.add_argument('--t2', type=str, default=None, help='Path to single T2 image')
    parser.add_argument('--t1_dir', type=str, default=None, help='Directory containing T1 images')
    parser.add_argument('--t2_dir', type=str, default=None, help='Directory containing T2 images')
    
    # Output Parameters
    parser.add_argument('--output_dir', type=str, default='inference_outputs',
                        help='Directory to save prediction masks and visualizations')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for change classification (default: 0.5)')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image input resolution (default: 256)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_overlay', action='store_true', default=True,
                        help='Save red highlighted change overlay on T2 image')
    return parser.parse_args()


def preprocess_image(img_path, img_size=(256, 256)):
    """Loads and normalizes an image for TinyCD input."""
    img = Image.open(img_path).convert('RGB')
    orig_size = img.size # (W, H)
    img_resized = TF.resize(img, img_size, interpolation=TF.InterpolationMode.BILINEAR)
    tensor = TF.to_tensor(img_resized)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return tensor.unsqueeze(0), img, orig_size


def create_change_overlay(t2_pil, binary_mask, alpha=0.45):
    """Creates a visually clear semi-transparent red overlay on top of T2."""
    t2_np = np.array(t2_pil).astype(np.float32)
    mask_np = np.array(binary_mask) > 127

    overlay = t2_np.copy()
    # Apply red mask highlight: [255, 30, 30]
    overlay[mask_np, 0] = (1 - alpha) * overlay[mask_np, 0] + alpha * 255
    overlay[mask_np, 1] = (1 - alpha) * overlay[mask_np, 1] + alpha * 30
    overlay[mask_np, 2] = (1 - alpha) * overlay[mask_np, 2] + alpha * 30

    return np.clip(overlay, 0, 255).astype(np.uint8)


@torch.no_grad()
def run_single_inference(model, t1_path, t2_path, output_dir, device, threshold=0.5, img_size=(256, 256), save_overlay=True):
    t1_tensor, t1_orig_pil, orig_size = preprocess_image(t1_path, img_size)
    t2_tensor, t2_orig_pil, _ = preprocess_image(t2_path, img_size)

    t1_tensor = t1_tensor.to(device)
    t2_tensor = t2_tensor.to(device)

    logits = model(t1_tensor, t2_tensor)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Threshold to binary change mask (0 or 255)
    binary_mask_256 = (probs > threshold).astype(np.uint8) * 255
    mask_pil = Image.fromarray(binary_mask_256).resize(orig_size, resample=Image.NEAREST)

    base_name = os.path.splitext(os.path.basename(t1_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save raw binary mask
    mask_save_path = os.path.join(output_dir, f"{base_name}_pred_mask.png")
    mask_pil.save(mask_save_path)

    # 2. Save Overlay and 3-Panel Inspection Figure
    if save_overlay:
        overlay_np = create_change_overlay(t2_orig_pil, mask_pil)
        overlay_save_path = os.path.join(output_dir, f"{base_name}_overlay.png")
        Image.fromarray(overlay_np).save(overlay_save_path)

        # 3-Panel Comparison Figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
        axes[0].imshow(t1_orig_pil)
        axes[0].set_title("Before Image ($T_1$)", fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(t2_orig_pil)
        axes[1].set_title("After Image ($T_2$)", fontsize=12)
        axes[1].axis('off')

        axes[2].imshow(overlay_np)
        axes[2].set_title("Detected Changes (Red Overlay)", fontsize=12, fontweight='bold', color='crimson')
        axes[2].axis('off')

        plt.tight_layout()
        comparison_save_path = os.path.join(output_dir, f"{base_name}_comparison.png")
        plt.savefig(comparison_save_path, bbox_inches='tight')
        plt.close()

    print(f"✅ Processed: {base_name} -> Mask saved to {mask_save_path}")


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load Model
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    saved_args = checkpoint.get('args', {})
    backbone = saved_args.get('backbone', 'efficientnet_b0')

    model = TinyCD(backbone_name=backbone, pretrained=False, num_classes=1).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    if args.t1 and args.t2:
        # Single Pair Inference
        run_single_inference(
            model=model,
            t1_path=args.t1,
            t2_path=args.t2,
            output_dir=args.output_dir,
            device=device,
            threshold=args.threshold,
            img_size=(args.img_size, args.img_size),
            save_overlay=args.save_overlay
        )
    elif args.t1_dir and args.t2_dir:
        # Batch Folder Inference
        exts = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff')
        t1_files = []
        for ext in exts:
            t1_files.extend(glob.glob(os.path.join(args.t1_dir, ext)))
        
        print(f"Found {len(t1_files)} images to process in {args.t1_dir}...")
        for f1 in t1_files:
            bname = os.path.basename(f1)
            f2 = os.path.join(args.t2_dir, bname)
            if not os.path.exists(f2):
                matches = glob.glob(os.path.join(args.t2_dir, f"{os.path.splitext(bname)[0]}.*"))
                if matches:
                    f2 = matches[0]

            if os.path.exists(f2):
                run_single_inference(
                    model=model,
                    t1_path=f1,
                    t2_path=f2,
                    output_dir=args.output_dir,
                    device=device,
                    threshold=args.threshold,
                    img_size=(args.img_size, args.img_size),
                    save_overlay=args.save_overlay
                )
            else:
                print(f"⚠️ Warning: Missing corresponding T2 image for {bname}")
    else:
        print("❌ Error: Please provide either (--t1 and --t2) for single pair or (--t1_dir and --t2_dir) for folder inference.")


if __name__ == '__main__':
    main()
