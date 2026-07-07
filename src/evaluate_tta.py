import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from tqdm import tqdm

from config import load_config, ensure_dirs
from data import build_train_df, WasteDataset
from transforms import get_valid_transforms
from model import create_model
from utils import seed_everything, get_device


def make_loader(dataset, batch_size, cfg):
    num_workers = int(cfg.get('num_workers', 2))
    kwargs = dict(
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(cfg.get('pin_memory', True)),
    )
    if num_workers > 0:
        kwargs['persistent_workers'] = bool(cfg.get('persistent_workers', True))
        kwargs['prefetch_factor'] = int(cfg.get('prefetch_factor', 2))
    return DataLoader(dataset, **kwargs)


def save_confusion_matrix_png(cm, path, class_names):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=35, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Validation Confusion Matrix')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def load_model_state(ckpt_path, device):
    """Support both raw state_dict and full checkpoint dict."""
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        return ckpt['model'], ckpt
    return ckpt, {'model': ckpt}


@torch.no_grad()
def predict_val(model, loader, device, cfg, tta_hflip=False):
    model.eval()
    all_probs, all_labels, all_paths = [], [], []
    use_amp = bool(cfg.get('use_amp', True)) and device.type == 'cuda'

    for images, labels, paths in tqdm(loader, desc='Evaluate val'):
        images = images.to(device, non_blocking=True)
        if cfg.get('channels_last', True):
            images = images.contiguous(memory_format=torch.channels_last)

        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            if tta_hflip:
                flipped = torch.flip(images, dims=[3])
                logits_flip = model(flipped)
                probs_flip = torch.softmax(logits_flip, dim=1)
                probs = (probs + probs_flip) / 2.0

        all_probs.append(probs.detach().cpu().numpy())
        all_labels.extend(labels.numpy().tolist())
        all_paths.extend(list(paths))

    probs = np.concatenate(all_probs, axis=0)
    preds = probs.argmax(axis=1)
    return np.array(all_labels), preds, probs, all_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', default=None, help='Optional checkpoint path. Default: weights/<run_name>/best.pth')
    parser.add_argument('--tta-hflip', action='store_true', help='Use horizontal flip TTA during validation evaluation')
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    seed_everything(int(cfg.get('seed', 42)))

    device = get_device()
    print('Device:', device)
    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0))

    df = build_train_df(cfg['data_dir'], cfg['train_dir'], cfg['class_folders'])
    _, val_df = train_test_split(
        df,
        test_size=float(cfg.get('val_size', 0.2)),
        stratify=df['label'],
        random_state=int(cfg.get('seed', 42)),
    )
    val_df = val_df.reset_index(drop=True)
    print('Validation images:', len(val_df))
    print('TTA horizontal flip:', args.tta_hflip)

    label_to_name = {int(v): k for k, v in cfg['class_folders'].items()}
    class_names = [label_to_name[i].split('_', 1)[-1] for i in range(int(cfg.get('num_classes', 3)))]

    ds = WasteDataset(val_df, transform=get_valid_transforms(cfg), return_path=True)
    loader = make_loader(ds, int(cfg.get('val_batch_size', cfg.get('batch_size', 8))), cfg)

    model = create_model({**cfg, 'pretrained': False}).to(device)
    if cfg.get('channels_last', True) and device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else Path(cfg.get('output_dir', '.')) / cfg['weights_dir'] / cfg.get('run_name', 'run') / cfg.get('checkpoint_name', 'best.pth')
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    state, ckpt = load_model_state(ckpt_path, device)
    model.load_state_dict(state)
    print('Loaded checkpoint:', ckpt_path)
    if isinstance(ckpt, dict):
        if 'epoch' in ckpt:
            print('Checkpoint epoch:', ckpt['epoch'])
        if 'best_f1' in ckpt:
            print('Checkpoint best_f1:', ckpt['best_f1'])

    y_true, y_pred, probs, paths = predict_val(model, loader, device, cfg, tta_hflip=args.tta_hflip)

    macro = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    suffix = 'evaluation_tta_hflip' if args.tta_hflip else 'evaluation'
    out_dir = Path(cfg.get('output_dir', '.')) / cfg['results_dir'] / cfg.get('run_name', 'run') / suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    summary.append(f'Checkpoint: {ckpt_path}')
    summary.append(f'TTA horizontal flip: {args.tta_hflip}')
    summary.append(f'Accuracy: {acc:.6f}')
    summary.append(f'Macro F1: {macro:.6f}')
    summary.append('')
    summary.append(report)
    summary_text = '\n'.join(summary)

    print(summary_text)
    (out_dir / 'classification_report.txt').write_text(summary_text, encoding='utf-8')
    np.savetxt(out_dir / 'confusion_matrix.csv', cm, fmt='%d', delimiter=',')
    save_confusion_matrix_png(cm, out_dir / 'confusion_matrix.png', class_names)

    pred_df = pd.DataFrame({
        'path': paths,
        'label': y_true,
        'pred': y_pred,
        'is_correct': y_true == y_pred,
    })

    for i, name in enumerate(class_names):
        pred_df[f'prob_{i}_{name}'] = probs[:, i]

    sorted_probs = np.sort(probs, axis=1)
    pred_df['confidence'] = probs.max(axis=1)
    pred_df['top2_gap'] = sorted_probs[:, -1] - sorted_probs[:, -2]

    pred_df.to_csv(out_dir / 'val_predictions.csv', index=False)
    pred_df[pred_df['is_correct'] == False].to_csv(out_dir / 'wrong_val_predictions.csv', index=False)
    pred_df[pred_df['is_correct'] == True].to_csv(out_dir / 'correct_val_predictions.csv', index=False)

    # Confidence histogram
    plt.figure(figsize=(7, 4))
    plt.hist(pred_df['confidence'], bins=50)
    plt.title('Validation Confidence Histogram')
    plt.xlabel('Max probability')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(out_dir / 'confidence_histogram.png', dpi=180)
    plt.close()

    # Top-2 gap histogram
    plt.figure(figsize=(7, 4))
    plt.hist(pred_df['top2_gap'], bins=50)
    plt.title('Validation Top-2 Probability Gap')
    plt.xlabel('Top1 prob - Top2 prob')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(out_dir / 'top2_gap_histogram.png', dpi=180)
    plt.close()

    print('Saved evaluation to:', out_dir)
    print('Confusion matrix rows=true, cols=pred:')
    print(cm)


if __name__ == '__main__':
    main()
