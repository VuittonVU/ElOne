import timm


def create_model(cfg):
    return timm.create_model(
        cfg.get('model_name', 'convnext_base'),
        pretrained=bool(cfg.get('pretrained', True)),
        num_classes=int(cfg.get('num_classes', 3)),
    )
