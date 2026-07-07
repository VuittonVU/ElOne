import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(cfg):
    size = int(cfg['image_size'])
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=float(cfg.get('hflip_prob', 0.5))),
        A.RandomBrightnessContrast(p=float(cfg.get('brightness_contrast_prob', 0.25))),
        A.CoarseDropout(
            num_holes_range=(1, 2),
            hole_height_range=(0.04, 0.10),
            hole_width_range=(0.04, 0.10),
            fill=0,
            p=float(cfg.get('random_erasing_prob', 0.1)),
        ),
        A.Normalize(),
        ToTensorV2(),
    ])


def get_valid_transforms(cfg):
    size = int(cfg['image_size'])
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(),
        ToTensorV2(),
    ])
