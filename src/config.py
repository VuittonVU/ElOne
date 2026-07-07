from pathlib import Path
import yaml


def load_config(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg


def project_path(cfg, key: str):
    output_dir = Path(cfg.get('output_dir', '.'))
    return output_dir / cfg[key]


def ensure_dirs(cfg):
    for key in ['weights_dir', 'logs_dir', 'results_dir', 'submissions_dir']:
        Path(cfg.get('output_dir', '.')).joinpath(cfg[key]).mkdir(parents=True, exist_ok=True)
