import os
import argparse
import torch
from tqdm import tqdm

from models import TinyCD
from datasets import build_dataloader
from utils import MetricEvaluator, save_comparison_figure


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TinyCD on SYSU-CD Test Set (SIH PS 1518)")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the root directory of SYSU-CD dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint (e.g., checkpoints/best_model.pth)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['test', 'val', 'train'],
                        help='Dataset split to evaluate on (default: test)')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution (default: 256)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_visuals', action='store_true', default=True,
                        help='Save side-by-side visual comparison images for test samples')
    parser.add_argument('--visuals_dir', type=str, default='eval_results/visuals',
                        help='Directory to save visual comparison outputs')
    parser.add_argument('--max_visuals', type=int, default=50,
                        help='Maximum number of visual comparison figures to save')
    return parser.parse_args()


@torch.no_grad()
def evaluate():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 75)
    print(f"📊 Evaluating TinyCD on SYSU-CD [{args.split.upper()}] Set")
    print("=" * 75)
    print(f"📁 Dataset Directory: {args.data_dir}")
    print(f"💾 Checkpoint:        {args.checkpoint}")
    print(f"⚡ Device:            {device}")
    print("=" * 75)

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")

    # Load Checkpoint & Hyperparameters
    checkpoint = torch.load(args.checkpoint, map_location=device)
    saved_args = checkpoint.get('args', {})
    backbone = saved_args.get('backbone', 'efficientnet_b0')

    # Load Model
    model = TinyCD(backbone_name=backbone, pretrained=False, num_classes=1).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"✅ Model weights loaded successfully (Backbone: {backbone})")

    # Build Test DataLoader
    test_loader, test_ds = build_dataloader(
        root_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        img_size=(args.img_size, args.img_size)
    )
    print(f"✅ Total {args.split.upper()} Samples: {len(test_ds)}")

    if args.save_visuals:
        os.makedirs(args.visuals_dir, exist_ok=True)

    metric_evaluator = MetricEvaluator()
    saved_visual_count = 0

    pbar = tqdm(test_loader, desc="Evaluating", dynamic_ncols=True)
    for batch in pbar:
        t1 = batch['t1'].to(device, non_blocking=True)
        t2 = batch['t2'].to(device, non_blocking=True)
        targets = batch['mask'].to(device, non_blocking=True)
        names = batch['name']

        logits = model(t1, t2)
        metric_evaluator.update(logits, targets)

        # Save visual inspections
        if args.save_visuals and saved_visual_count < args.max_visuals:
            for i in range(t1.size(0)):
                if saved_visual_count >= args.max_visuals:
                    break
                sample_name = names[i]
                save_path = os.path.join(args.visuals_dir, f"{sample_name}_comparison.png")
                save_comparison_figure(
                    t1=t1[i],
                    t2=t2[i],
                    gt=targets[i],
                    pred=logits[i],
                    save_path=save_path,
                    sample_name=sample_name
                )
                saved_visual_count += 1

    # Fetch Final Quantitative Results
    m = metric_evaluator.get_metrics()

    print("\n" + "=" * 75)
    print("🏆 FINAL QUANTITATIVE EVALUATION RESULTS (SYSU-CD)")
    print("=" * 75)
    print(f"  • Change F1-Score (Dice) : {m['F1']:.2f} %")
    print(f"  • Change IoU (mIoU)      : {m['IoU']:.2f} %")
    print(f"  • Precision              : {m['Precision']:.2f} %")
    print(f"  • Recall                 : {m['Recall']:.2f} %")
    print(f"  • Overall Accuracy (OA)  : {m['OA']:.2f} %")
    print(f"  • Cohen's Kappa          : {m['Kappa']:.2f} %")
    print("-" * 75)
    print("  • Pixel Confusion Matrix:")
    print(f"    - True Positives (TP)  : {m['TP']:,}")
    print(f"    - False Positives (FP) : {m['FP']:,}")
    print(f"    - False Negatives (FN) : {m['FN']:,}")
    print(f"    - True Negatives (TN)  : {m['TN']:,}")
    print("=" * 75)

    if args.save_visuals:
        print(f"📁 Visual inspection figures saved to: {os.path.abspath(args.visuals_dir)}")
    print("=" * 75)


if __name__ == '__main__':
    evaluate()
