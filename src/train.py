# src/train.py
"""
Train script for Voyex Facial Recognition v1
Saves checkpoints with key 'model_state' and 'config' to be compatible with infer/centroid scripts.

Usage:
  python src/train.py --config config.yaml
"""
import os
import time
import argparse
import yaml
import random
import math
import shutil
from collections import deque
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import FolderDataset, get_train_transform, get_valid_transform
from model import ResNetEmbedding

# -------- utilities --------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def seed_everything(seed=42):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_checkpoint(state, path):
    torch.save(state, path)

# -------- lr scheduler: warmup + cosine --------
def build_lr_lambda(total_steps, warmup_steps):
    def _lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return _lr_lambda

# -------- training / validation --------
def train_one_epoch(model, loader, criterion, optimizer, device, scaler, epoch, cfg, writer=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Train E{epoch}")
    for i, (imgs, labels, _) in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(imgs)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

        if writer is not None:
            step = epoch * len(loader) + i
            writer.add_scalar("train/loss_step", loss.item(), step)

        pbar.set_postfix(loss=running_loss/total, acc=correct/total)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels, _ in tqdm(loader, desc="Validate", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    return running_loss / total, correct / total

# -------- main --------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Create necessary directories
    checkpoint_dir = cfg['checkpoint']['save_dir']
    log_dir = cfg['logging']['log_dir']
    tb_dir = cfg['logging']['tensorboard_dir']
    
    ensure_dir(checkpoint_dir)
    ensure_dir(log_dir)
    ensure_dir(tb_dir)

    # Set random seed and device
    seed_everything(cfg.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.get('device', 'cuda') == 'cuda' else 'cpu')

    # dataset and loaders
    img_size = cfg['data']['img_size']
    batch_size = cfg['data']['batch_size']
    num_workers = cfg['data'].get('num_workers', 4)
    
    train_transform = get_train_transform(img_size, cfg.get('random_erasing_p', 0.2))
    valid_transform = get_valid_transform(img_size)

    train_ds = FolderDataset(cfg['data']['train_dir'], transform=train_transform)
    val_ds = FolderDataset(cfg['data']['val_dir'], transform=valid_transform)

    num_classes = len(train_ds.classes)
    cfg['model']['num_classes'] = num_classes  # Update num_classes in config
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Model initialization
    model_cfg = cfg['model']
    model = ResNetEmbedding(
        backbone=model_cfg.get('backbone', 'resnet50'),
        emb_size=model_cfg.get('embedding_size', 512),
        num_classes=num_classes,
        pretrained=model_cfg.get('pretrained', True)
    )
    model = model.to(device)
    
    # Training setup
    train_cfg = cfg['training']
    
    # Criterion with label smoothing
    label_smoothing = train_cfg.get('label_smoothing', 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(train_cfg['lr']),
        weight_decay=float(train_cfg['weight_decay'])
    )
    
    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * train_cfg['epochs']
    warmup_steps = cfg['scheduler']['warmup_epochs'] * len(train_loader)
    lr_lambda = build_lr_lambda(total_steps, warmup_steps)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training parameters
    epochs = train_cfg['epochs']
    save_every = cfg['checkpoint']['save_freq']
    use_amp = cfg.get('use_amp', True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=tb_dir) if not cfg.get('debug', False) else None
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(epochs):
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, 
            device, scaler, epoch, cfg, writer
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, device)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        print(f'  LR: {current_lr:.2e}')
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'val_acc': val_acc,
                'config': cfg
            }
            
            # Save latest
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'latest.pth'))
            
            # Save best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(checkpoint, os.path.join(checkpoint_dir, 'best.pth'))
                print(f'New best model saved with accuracy: {best_val_acc:.2f}%')
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
    
    print(f'Training complete. Best validation accuracy: {best_val_acc:.2f}%')
    
    # Save final training log
    log_dir = cfg['logging']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    csv_log = os.path.join(log_dir, 'training_log.csv')
    
    # Write CSV header if file doesn't exist
    if not os.path.exists(csv_log):
        with open(csv_log, 'w') as f:
            f.write('epoch,train_loss,train_acc,val_loss,val_acc,lr\n')
    
    print(f"\nTraining completed! Checkpoints saved to: {checkpoint_dir}")
    print(f"Training log saved to: {csv_log}")


if __name__ == "__main__":
    main()
