from pathlib import Path
import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

VALID_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def build_train_df(data_dir, train_dir, class_folders):
    data_dir = Path(data_dir)
    train_path = data_dir / train_dir
    if not train_path.exists():
        raise FileNotFoundError(f'Train folder not found: {train_path}')

    rows = []
    missing = []
    for folder_name, label in class_folders.items():
        folder = train_path / folder_name
        if not folder.exists():
            missing.append(str(folder))
            continue
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix.lower() in VALID_EXTS:
                rows.append({
                    'path': str(p),
                    'filename': p.name,
                    'class': folder_name,
                    'label': int(label),
                })
    if missing:
        raise FileNotFoundError('Class folder(s) not found:\n' + '\n'.join(missing))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f'No images found in {train_path}')
    return df


class WasteDataset(Dataset):
    def __init__(self, df, transform=None, return_path=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.return_path = return_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = np.array(Image.open(row["path"]).convert("RGB"))
        label = int(row["label"])

        if self.transform:
            image = self.transform(image=image)["image"]

        if self.return_path:
            return image, label, row["path"]

        return image, label


class TestDataset(Dataset):
    def __init__(self, template_df, test_dir, transform=None):
        self.df = template_df.reset_index(drop=True)
        self.test_dir = Path(test_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _resolve_image_path(self, img_id):
        # template id can be 1, "1", "1.jpg", etc.
        s = str(img_id)
        candidates = []
        if Path(s).suffix.lower() in VALID_EXTS:
            candidates.append(self.test_dir / s)
        else:
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                candidates.append(self.test_dir / f'{s}{ext}')
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(f'Test image not found for id={img_id}. Tried: {candidates[:3]} ...')

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_id = str(int(float(row['id'])))

        candidates = [
            os.path.join(self.test_dir, f"{img_id}.jpg"),
            os.path.join(self.test_dir, f"{img_id}.jpeg"),
            os.path.join(self.test_dir, f"{img_id}.png"),
        ]

        path = None
        for p in candidates:
            if os.path.exists(p):
                path = p
                break

        if path is None:
            raise FileNotFoundError(f"Cannot find image for id={img_id}")

        image = np.array(Image.open(path).convert("RGB"))

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, int(img_id)
