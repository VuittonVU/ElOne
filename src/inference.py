import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import load_config, ensure_dirs
from data import TestDataset
from transforms import get_valid_transforms
from model import create_model
from utils import get_device


@torch.no_grad()
def predict_loader(model, loader, device, cfg, tta_hflip=False):
    model.eval()
    all_probs = []
    all_ids = []
    use_amp = bool(cfg.get('use_amp', True)) and device.type == 'cuda'
    for images, img_ids in tqdm(loader, desc='Predict'):
        images = images.to(device, non_blocking=True)
        if cfg.get('channels_last', True):
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            if tta_hflip:
                logits_flip = model(torch.flip(images, dims=[3]))
                probs = (probs + torch.softmax(logits_flip, dim=1)) / 2.0
        all_probs.append(probs.detach().cpu().numpy())
        if hasattr(img_ids, 'numpy'):
            all_ids.extend(img_ids.numpy().tolist())
        else:
            all_ids.extend(list(img_ids))
    return all_ids, np.concatenate(all_probs, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    device = get_device()
    print('Device:', device)
    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0))

    data_dir = Path(cfg['data_dir'])
    template_path = data_dir / cfg['submission_file']
    test_dir = data_dir / cfg['test_dir']
    template = pd.read_csv(template_path)
    if 'id' not in template.columns:
        raise ValueError(f'Template must contain id column. Columns found: {template.columns.tolist()}')

    test_ds = TestDataset(template, test_dir, get_valid_transforms(cfg))
    loader = DataLoader(test_ds, batch_size=int(cfg.get('batch_size', 8)), shuffle=False, num_workers=int(cfg.get('num_workers', 2)), pin_memory=bool(cfg.get('pin_memory', True)))

    model = create_model({**cfg, 'pretrained': False}).to(device)
    if cfg.get('channels_last', True) and device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)

    ckpt_path = Path(cfg.get('output_dir', '.')) / cfg['weights_dir'] / cfg.get('run_name', 'run') / cfg.get('checkpoint_name', 'best.pth')
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}. Train first.')
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    print('Loaded checkpoint:', ckpt_path)

    ids, probs = predict_loader(model, loader, device, cfg, tta_hflip=bool(cfg.get('tta_hflip', True)))
    preds = probs.argmax(axis=1).astype(int)

    submission = template.copy()
    submission['predicted'] = preds
    out_dir = Path(cfg.get('output_dir', '.')) / cfg['submissions_dir']
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg.get('submission_name', 'submission.csv')
    submission.to_csv(out_path, index=False)

    print('Saved submission:', out_path)
    print(submission.head())
    print(submission['predicted'].value_counts())


if __name__ == '__main__':
    main()
