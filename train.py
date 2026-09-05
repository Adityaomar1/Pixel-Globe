import os
import sys
import time
import argparse
import csv
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from models import TinyCD, CombinedLoss, count_parameters
from datasets import build_dataloader
from utils import MetricEvaluator, save_comparison_figure


def parse_args():
    parser = argparse.ArgumentParser(description="Train TinyCD for Change Detection on SYSU-CD Dataset (SIH PS 1518)")
    
    # Dataset Parameters
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the root directory of SYSU-CD dataset')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Input image resolution (default: 256 for 256x256)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader worker processes')
    
    # Model Architecture
    parser.add_argument('--backbone', type=str, default='efficientnet_b0',
                        choices=['efficientnet_b0', 'efficientnet_b4'],
                        help='Backbone architecture for Siamese feature extractor (default: efficientnet_b0)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use ImageNet pretrained weights for backbone')
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=60,
                        help='Number of training epochs (default: 60)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Mini-batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Base learning rate (default: 5e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for AdamW optimizer')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate for CosineAnnealingLR')
    
    # Loss Weights
    parser.add_argument('--w_bce', type=float, default=1.0, help='Weight for BCE loss')
    parser.add_argument('--w_dice', type=float, default=1.0, help='Weight for Dice loss')
    parser.add_argument('--w_focal', type=float, default=0.5, help='Weight for Focal loss')
    
    # Environment & Hardware
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Computation device (cuda or cpu)')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Use Automatic Mixed Precision (AMP fp16) for 2x faster training and reduced VRAM')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save trained model checkpoints and training logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to existing checkpoint to resume training from')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience in epochs')

    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    running_loss = 0.0
    metric_evaluator = MetricEvaluator()

    pbar = tqdm(dataloader, desc="Train", leave=False, dynamic_ncols=True)
    for batch in pbar:
        t1 = batch['t1'].to(device, non_blocking=True)
        t2 = batch['t2'].to(device, non_blocking=True)
        targets = batch['mask'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == 'cuda':
            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(t1, t2)
                loss, _ = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(t1, t2)
            loss, _ = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        metric_evaluator.update(logits.detach(), targets.detach())
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    epoch_loss = running_loss / max(1, len(dataloader))
    metrics = metric_evaluator.get_metrics()
    metrics['loss'] = epoch_loss
    return metrics


@torch.no_grad()
def validate(model, dataloader, criterion, device, use_amp, save_visuals_dir=None, epoch=0):
    model.eval()
    running_loss = 0.0
    metric_evaluator = MetricEvaluator()
    saved_visual = False

    pbar = tqdm(dataloader, desc="Val", leave=False, dynamic_ncols=True)
    for batch in pbar:
        t1 = batch['t1'].to(device, non_blocking=True)
        t2 = batch['t2'].to(device, non_blocking=True)
        targets = batch['mask'].to(device, non_blocking=True)

        if use_amp and device.type == 'cuda':
            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(t1, t2)
                loss, _ = criterion(logits, targets)
        else:
            logits = model(t1, t2)
            loss, _ = criterion(logits, targets)

        running_loss += loss.item()
        metric_evaluator.update(logits, targets)

        # Save first batch sample visualization for inspection
        if save_visuals_dir and not saved_visual:
            sample_path = os.path.join(save_visuals_dir, f"val_sample_epoch_{epoch:03d}.png")
            save_comparison_figure(
                t1=t1[0],
                t2=t2[0],
                gt=targets[0],
                pred=logits[0],
                save_path=sample_path,
                sample_name=f"Epoch {epoch} - {batch['name'][0]}"
            )
            saved_visual = True

    epoch_loss = running_loss / max(1, len(dataloader))
    metrics = metric_evaluator.get_metrics()
    metrics['loss'] = epoch_loss
    return metrics


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    visuals_dir = os.path.join(args.save_dir, "visuals")
    os.makedirs(visuals_dir, exist_ok=True)

    print("=" * 75)
    print("🚀 SIH PS 1518: TinyCD Training on SYSU-CD Dataset")
    print("=" * 75)
    print(f"📁 Dataset Directory: {args.data_dir}")
    print(f"⚙️ Backbone:          {args.backbone} (Pretrained={args.pretrained})")
    print(f"🖼️ Image Resolution:  {args.img_size}x{args.img_size}")
    print(f"📦 Batch Size:        {args.batch_size}")
    print(f"⚡ Device:            {device} (AMP={args.amp and device.type == 'cuda'})")
    print(f"🎯 Max Epochs:        {args.epochs}")
    print("=" * 75)

    # 1. Build DataLoaders
    print("\n[1/5] Initializing SYSU-CD DataLoaders...")
    train_loader, train_ds = build_dataloader(
        root_dir=args.data_dir, split='train',
        batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, img_size=(args.img_size, args.img_size)
    )
    print(f"   -> Train samples: {len(train_ds)} ({len(train_loader)} batches)")

    try:
        val_loader, val_ds = build_dataloader(
            root_dir=args.data_dir, split='val',
            batch_size=args.batch_size, num_workers=args.num_workers,
            shuffle=False, img_size=(args.img_size, args.img_size)
        )
        print(f"   -> Val samples:   {len(val_ds)} ({len(val_loader)} batches)")
    except Exception as e:
        print(f"   ⚠️ Validation split error ({e}), falling back to test split for validation...")
        val_loader, val_ds = build_dataloader(
            root_dir=args.data_dir, split='test',
            batch_size=args.batch_size, num_workers=args.num_workers,
            shuffle=False, img_size=(args.img_size, args.img_size)
        )
        print(f"   -> Val samples (from test): {len(val_ds)} ({len(val_loader)} batches)")

    # 2. Build Model
    print("\n[2/5] Initializing TinyCD Architecture...")
    model = TinyCD(backbone_name=args.backbone, pretrained=args.pretrained, num_classes=1).to(device)
    params_count = count_parameters(model)
    print(f"   -> Total Trainable Parameters: {params_count:.2f} Million (Super Lightweight!)")

    # 3. Setup Loss, Optimizer, Scheduler, AMP Scaler
    criterion = CombinedLoss(w_bce=args.w_bce, w_dice=args.w_dice, w_focal=args.w_focal)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scaler = GradScaler('cuda', enabled=(args.amp and device.type == 'cuda'))

    start_epoch = 1
    best_f1 = 0.0
    best_iou = 0.0
    patience_counter = 0

    # Resume if requested
    if args.resume and os.path.isfile(args.resume):
        print(f"\n[!] Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        if 'scheduler_state' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint.get('best_f1', 0.0)
        best_iou = checkpoint.get('best_iou', 0.0)
        print(f"   -> Loaded epoch {checkpoint['epoch']} with best F1: {best_f1:.2f}%")

    # Prepare CSV Logger
    log_file_path = os.path.join(args.save_dir, "training_log.csv")
    is_new_log = not os.path.exists(log_file_path) or start_epoch == 1
    log_fields = ['epoch', 'train_loss', 'train_f1', 'train_iou', 'val_loss', 'val_f1', 'val_iou', 'val_precision', 'val_recall', 'val_oa', 'val_kappa', 'lr']

    if is_new_log:
        with open(log_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(log_fields)

    print("\n[3/5] Starting Training Loop...")
    total_start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        # Train Step
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=args.amp
        )

        # Validation Step
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=args.amp,
            save_visuals_dir=visuals_dir,
            epoch=epoch
        )

        scheduler.step()
        epoch_duration = time.time() - epoch_start_time

        # Print Epoch Summary
        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] ({epoch_duration:.1f}s, LR: {current_lr:.6f}) | "
            f"Train Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['F1']:.2f}%, IoU: {train_metrics['IoU']:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['F1']:.2f}%, IoU: {val_metrics['IoU']:.2f}%, OA: {val_metrics['OA']:.2f}%"
        )

        # Log to CSV
        with open(log_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_metrics['loss']:.4f}", f"{train_metrics['F1']:.2f}", f"{train_metrics['IoU']:.2f}",
                f"{val_metrics['loss']:.4f}", f"{val_metrics['F1']:.2f}", f"{val_metrics['IoU']:.2f}",
                f"{val_metrics['Precision']:.2f}", f"{val_metrics['Recall']:.2f}", f"{val_metrics['OA']:.2f}", f"{val_metrics['Kappa']:.2f}",
                f"{current_lr:.6f}"
            ])

        # Checkpoint Saving
        checkpoint_data = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_f1': best_f1,
            'best_iou': best_iou,
            'args': vars(args)
        }

        # Save latest model
        torch.save(checkpoint_data, os.path.join(args.save_dir, "latest_model.pth"))

        # Save best model based on F1-Score
        if val_metrics['F1'] > best_f1:
            best_f1 = val_metrics['F1']
            best_iou = val_metrics['IoU']
            patience_counter = 0
            checkpoint_data['best_f1'] = best_f1
            checkpoint_data['best_iou'] = best_iou
            torch.save(checkpoint_data, os.path.join(args.save_dir, "best_model.pth"))
            print(f"   🌟 New Best Model Saved! [Val F1: {best_f1:.2f}%, Val IoU: {best_iou:.2f}%]")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⚠️ Early stopping triggered: Validation F1 did not improve for {args.patience} consecutive epochs.")
                break

    total_time_min = (time.time() - total_start_time) / 60.0
    print("\n" + "=" * 75)
    print(f"🎉 Training Finished in {total_time_min:.2f} minutes!")
    print(f"🏆 Best Validation F1-Score: {best_f1:.2f}% | Best IoU: {best_iou:.2f}%")
    print(f"💾 Best Model Weights: {os.path.join(args.save_dir, 'best_model.pth')}")
    print(f"📊 Training Log:       {log_file_path}")
    print("=" * 75)


if __name__ == '__main__':
    main()
