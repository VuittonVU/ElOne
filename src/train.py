import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import load_config, ensure_dirs
from data import build_train_df, WasteDataset
from transforms import get_train_transforms, get_valid_transforms
from model import create_model
from metrics import macro_f1, make_report, make_confusion_matrix
from utils import seed_everything, get_device, format_seconds


def compute_class_weights(labels, num_classes, power=1.0):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    total = counts.sum()
    weights = total / (num_classes * counts)
    weights = weights ** float(power)
    weights = weights / weights.mean()
    return weights.astype(np.float32), counts.astype(int)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, cfg, epoch):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f'Epoch {epoch} Train')
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if cfg.get('channels_last', True):
            images = images.contiguous(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        use_amp = bool(cfg.get('use_amp', True)) and device.type == 'cuda'
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            if cfg.get('grad_clip_norm'):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg['grad_clip_norm']))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.get('grad_clip_norm'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg['grad_clip_norm']))
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f'{loss.item():.4f}')
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, criterion, device, cfg):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    use_amp = bool(cfg.get('use_amp', True)) and device.type == 'cuda'
    for images, labels in tqdm(loader, desc='Valid'):
        images = images.to(device, non_blocking=True)
        labels_gpu = labels.to(device, non_blocking=True)
        if cfg.get('channels_last', True):
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels_gpu)
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
        total_loss += loss.item()
    f1 = macro_f1(all_labels, all_preds)
    return total_loss / max(len(loader), 1), f1, np.array(all_labels), np.array(all_preds)


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_f1, cfg):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'scaler': scaler.state_dict() if scaler is not None else None,
        'epoch': epoch,
        'best_f1': best_f1,
        'cfg': cfg,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    if optimizer is not None and ckpt.get('optimizer') is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None and ckpt.get('scheduler') is not None:
        scheduler.load_state_dict(ckpt['scheduler'])
    if scaler is not None and ckpt.get('scaler') is not None:
        scaler.load_state_dict(ckpt['scaler'])
    return int(ckpt.get('epoch', 0)), float(ckpt.get('best_f1', 0.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    seed_everything(int(cfg.get('seed', 42)))

    device = get_device()
    print('Device:', device)
    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0))
    else:
        print('WARNING: CUDA not detected. Training on CPU will be very slow.')

    df = build_train_df(cfg['data_dir'], cfg['train_dir'], cfg['class_folders'])
    train_df, val_df = train_test_split(
        df,
        test_size=float(cfg.get('val_size', 0.2)),
        stratify=df['label'],
        random_state=int(cfg.get('seed', 42)),
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print('Train:', len(train_df), 'Val:', len(val_df))
    print('Train class counts:\n', train_df['class'].value_counts())
    print('Val class counts:\n', val_df['class'].value_counts())

    train_ds = WasteDataset(train_df, get_train_transforms(cfg))
    val_ds = WasteDataset(val_df, get_valid_transforms(cfg))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get('batch_size', 8)),
        shuffle=True,
        num_workers=int(cfg.get('num_workers', 2)),
        pin_memory=bool(cfg.get('pin_memory', True)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.get('batch_size', 8)),
        shuffle=False,
        num_workers=int(cfg.get('num_workers', 2)),
        pin_memory=bool(cfg.get('pin_memory', True)),
    )

    model = create_model(cfg).to(device)
    if cfg.get('channels_last', True) and device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)

    num_classes = int(cfg.get('num_classes', 3))
    if bool(cfg.get('use_class_weights', True)):
        weights, counts = compute_class_weights(train_df['label'].values, num_classes, cfg.get('class_weight_power', 1.0))
        print('Class counts:', counts.tolist())
        print('Class weights:', weights.tolist())
        weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    else:
        weight_tensor = None

    criterion = nn.CrossEntropyLoss(
        weight=weight_tensor,
        label_smoothing=float(cfg.get('label_smoothing', 0.0)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get('learning_rate', 1e-4)),
        weight_decay=float(cfg.get('weight_decay', 1e-4)),
    )
    scheduler = None
    if cfg.get('scheduler', 'cosine') == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg.get('epochs', 30)))

    scaler = torch.amp.GradScaler('cuda', enabled=(bool(cfg.get('use_amp', True)) and device.type == 'cuda'))

    out_dir = Path(cfg.get('output_dir', '.'))
    weights_dir = out_dir / cfg['weights_dir'] / cfg.get('run_name', 'run')
    logs_dir = out_dir / cfg['logs_dir'] / cfg.get('run_name', 'run')
    results_dir = out_dir / cfg['results_dir'] / cfg.get('run_name', 'run')
    weights_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    best_path = weights_dir / 'best.pth'
    last_path = weights_dir / 'last.pth'
    writer = SummaryWriter(log_dir=str(logs_dir))

    start_epoch = 1
    best_f1 = 0.0
    no_improve = 0
    if bool(cfg.get('resume', True)) and last_path.exists():
        loaded_epoch, best_f1 = load_checkpoint(last_path, model, optimizer, scheduler, scaler, device)
        start_epoch = loaded_epoch + 1
        print(f'Resumed from {last_path}, start_epoch={start_epoch}, best_f1={best_f1:.5f}')

    total_epochs = int(cfg.get('epochs', 30))
    max_epochs_per_run = cfg.get('max_epochs_per_run', None)
    time_limit_hours = cfg.get('time_limit_hours', None)
    run_start_epoch = start_epoch
    run_start_time = time.time()
    history_rows = []

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, cfg, epoch)
        val_loss, val_f1, y_true, y_pred = validate(model, val_loader, criterion, device, cfg)
        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch}/{total_epochs} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | val_macro_f1={val_f1:.5f} | lr={lr:.2e} | time={format_seconds(epoch_time)}')

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metric/macro_f1', val_f1, epoch)
        writer.add_scalar('LR', lr, epoch)
        history_rows.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'macro_f1': val_f1, 'lr': lr, 'epoch_time_sec': epoch_time})

        save_checkpoint(last_path, model, optimizer, scheduler, scaler, epoch, best_f1, cfg)
        improved = val_f1 > best_f1
        if improved:
            best_f1 = val_f1
            no_improve = 0
            save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch, best_f1, cfg)
            report = make_report(y_true, y_pred, target_names=['Recyclable', 'Electronic', 'Organic'])
            cm = make_confusion_matrix(y_true, y_pred)
            (results_dir / 'best_report.txt').write_text(report, encoding='utf-8')
            np.savetxt(results_dir / 'best_confusion_matrix.csv', cm, fmt='%d', delimiter=',')
            print('Saved best checkpoint:', best_path)
        else:
            no_improve += 1

        pd.DataFrame(history_rows).to_csv(results_dir / 'history_current_run.csv', index=False)

        if no_improve >= int(cfg.get('early_stopping_patience', 6)):
            print('Early stopping triggered.')
            break
        if max_epochs_per_run is not None and (epoch - run_start_epoch + 1) >= int(max_epochs_per_run):
            print(f'Stopping safely after max_epochs_per_run={max_epochs_per_run}. Run same command to resume.')
            break
        if time_limit_hours is not None and (time.time() - run_start_time) / 3600 >= float(time_limit_hours):
            print(f'Stopping safely after time_limit_hours={time_limit_hours}. Run same command to resume.')
            break

    writer.close()
    print('Training finished/safely paused.')
    print('Best F1:', best_f1)
    print('Best checkpoint:', best_path)
    print('Last checkpoint:', last_path)


if __name__ == '__main__':
    main()
