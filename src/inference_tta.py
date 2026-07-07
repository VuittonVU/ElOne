import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import load_config, ensure_dirs
from data import TestDataset
from transforms import get_valid_transforms
from model import create_model
from utils import get_device


def _bool_from_cfg_or_arg(cfg, key, cli_value, default=False):
    """CLI flag has priority. If CLI flag is None, use config value."""
    if cli_value is not None:
        return bool(cli_value)
    return bool(cfg.get(key, default))


def build_tta_transforms(cfg):
    """
    Build tensor-level TTA transforms.

    Safe default: original + horizontal flip.
    Optional extras can be enabled from config or CLI:
      tta_vflip: vertical flip
      tta_hvflip: horizontal + vertical flip
      tta_rot90: 90/180/270 degree rotations
    """
    transforms = [("orig", lambda x: x)]

    if bool(cfg.get("tta_hflip", True)):
        transforms.append(("hflip", lambda x: torch.flip(x, dims=[3])))

    if bool(cfg.get("tta_vflip", False)):
        transforms.append(("vflip", lambda x: torch.flip(x, dims=[2])))

    if bool(cfg.get("tta_hvflip", False)):
        transforms.append(("hvflip", lambda x: torch.flip(x, dims=[2, 3])))

    if bool(cfg.get("tta_rot90", False)):
        transforms.extend([
            ("rot90", lambda x: torch.rot90(x, k=1, dims=[2, 3])),
            ("rot180", lambda x: torch.rot90(x, k=2, dims=[2, 3])),
            ("rot270", lambda x: torch.rot90(x, k=3, dims=[2, 3])),
        ])

    return transforms


@torch.no_grad()
def predict_loader(model, loader, device, cfg):
    model.eval()
    all_probs = []
    all_ids = []
    use_amp = bool(cfg.get("use_amp", True)) and device.type == "cuda"
    tta_transforms = build_tta_transforms(cfg)
    print("TTA:", ", ".join(name for name, _ in tta_transforms))

    for images, img_ids in tqdm(loader, desc="Predict"):
        images = images.to(device, non_blocking=True)
        if cfg.get("channels_last", True):
            images = images.contiguous(memory_format=torch.channels_last)

        probs_sum = None
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            for _, transform_fn in tta_transforms:
                aug_images = transform_fn(images)
                if cfg.get("channels_last", True):
                    aug_images = aug_images.contiguous(memory_format=torch.channels_last)
                logits = model(aug_images)
                probs = torch.softmax(logits, dim=1)
                probs_sum = probs if probs_sum is None else probs_sum + probs

            probs = probs_sum / len(tta_transforms)

        all_probs.append(probs.detach().cpu().numpy())
        if hasattr(img_ids, "numpy"):
            all_ids.extend(img_ids.numpy().tolist())
        else:
            all_ids.extend(list(img_ids))

    return all_ids, np.concatenate(all_probs, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tta-none", action="store_true", help="Disable all TTA, use original image only.")
    parser.add_argument("--tta-hflip", action="store_true", default=None, help="Enable horizontal flip TTA.")
    parser.add_argument("--no-tta-hflip", dest="tta_hflip", action="store_false", help="Disable horizontal flip TTA.")
    parser.add_argument("--tta-vflip", action="store_true", default=None, help="Enable vertical flip TTA.")
    parser.add_argument("--tta-hvflip", action="store_true", default=None, help="Enable horizontal + vertical flip TTA.")
    parser.add_argument("--tta-rot90", action="store_true", default=None, help="Enable 90/180/270 rotation TTA.")
    parser.add_argument("--submission-name", default=None, help="Override output submission filename.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    if args.tta_none:
        cfg["tta_hflip"] = False
        cfg["tta_vflip"] = False
        cfg["tta_hvflip"] = False
        cfg["tta_rot90"] = False
    else:
        cfg["tta_hflip"] = _bool_from_cfg_or_arg(cfg, "tta_hflip", args.tta_hflip, default=True)
        cfg["tta_vflip"] = _bool_from_cfg_or_arg(cfg, "tta_vflip", args.tta_vflip, default=False)
        cfg["tta_hvflip"] = _bool_from_cfg_or_arg(cfg, "tta_hvflip", args.tta_hvflip, default=False)
        cfg["tta_rot90"] = _bool_from_cfg_or_arg(cfg, "tta_rot90", args.tta_rot90, default=False)

    if args.submission_name is not None:
        cfg["submission_name"] = args.submission_name

    device = get_device()
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    data_dir = Path(cfg["data_dir"])
    template_path = data_dir / cfg["submission_file"]
    test_dir = data_dir / cfg["test_dir"]
    template = pd.read_csv(template_path)
    if "id" not in template.columns:
        raise ValueError(f"Template must contain id column. Columns found: {template.columns.tolist()}")

    test_ds = TestDataset(template, test_dir, get_valid_transforms(cfg))
    loader = DataLoader(
        test_ds,
        batch_size=int(cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 2)),
        pin_memory=bool(cfg.get("pin_memory", True)),
    )

    model = create_model({**cfg, "pretrained": False}).to(device)
    if cfg.get("channels_last", True) and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    ckpt_path = Path(cfg.get("output_dir", ".")) / cfg["weights_dir"] / cfg.get("run_name", "run") / cfg.get("checkpoint_name", "best.pth")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Train first.")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    print("Loaded checkpoint:", ckpt_path)

    ids, probs = predict_loader(model, loader, device, cfg)
    preds = probs.argmax(axis=1).astype(int)

    submission = template.copy()
    submission["predicted"] = preds
    out_dir = Path(cfg.get("output_dir", ".")) / cfg["submissions_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg.get("submission_name", "submission.csv")
    submission.to_csv(out_path, index=False)

    print("Saved submission:", out_path)
    print(submission.head())
    print(submission["predicted"].value_counts())


if __name__ == "__main__":
    main()
