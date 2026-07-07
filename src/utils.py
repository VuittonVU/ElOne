import os
import random
import time
from pathlib import Path
import numpy as np
import torch


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def format_seconds(seconds: float):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f'{h}h {m}m {s}s'
    if m:
        return f'{m}m {s}s'
    return f'{s}s'


def save_text(path, text: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
