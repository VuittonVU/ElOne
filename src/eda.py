import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageStat
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import load_config, ensure_dirs
from data import build_train_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    df = build_train_df(cfg['data_dir'], cfg['train_dir'], cfg['class_folders'])
    print('Total train images:', len(df))
    print(df['class'].value_counts())

    rows, broken = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Scanning images'):
        try:
            img = Image.open(row['path']).convert('RGB')
            w, h = img.size
            stat = ImageStat.Stat(img)
            brightness = float(np.mean(stat.mean))
            rows.append({**row.to_dict(), 'width': w, 'height': h, 'aspect_ratio': w / h, 'brightness': brightness})
        except Exception as e:
            broken.append({'path': row['path'], 'error': str(e)})

    out_dir = Path(cfg.get('output_dir', '.'))
    results_dir = out_dir / cfg['results_dir']
    plots_dir = out_dir / 'plots'
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    info = pd.DataFrame(rows)
    info.to_csv(results_dir / 'eda_train_info.csv', index=False)
    pd.DataFrame(broken).to_csv(results_dir / 'broken_images.csv', index=False)

    summary = []
    summary.append(f'Total images: {len(df)}')
    summary.append(f'Valid images: {len(info)}')
    summary.append(f'Broken images: {len(broken)}')
    summary.append('\nClass distribution:')
    summary.append(str(info['class'].value_counts()))
    summary.append('\nResolution describe:')
    summary.append(str(info[['width', 'height', 'aspect_ratio', 'brightness']].describe()))
    text = '\n'.join(summary)
    print(text)
    (results_dir / 'eda_summary.txt').write_text(text, encoding='utf-8')

    plt.figure(figsize=(10, 6))
    info['class'].value_counts().plot(kind='bar')
    plt.title('Class Distribution')
    plt.tight_layout()
    plt.savefig(plots_dir / 'class_distribution.png', dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(info['width'], info['height'], alpha=0.25)
    plt.title('Image Width vs Height')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.tight_layout()
    plt.savefig(plots_dir / 'resolution_scatter.png', dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    info['brightness'].hist(bins=50)
    plt.title('Brightness Distribution')
    plt.tight_layout()
    plt.savefig(plots_dir / 'brightness_distribution.png', dpi=150)
    plt.close()

    print('EDA saved to:', results_dir)
    print('Plots saved to:', plots_dir)


if __name__ == '__main__':
    main()
