def build_backbone(cfg):
    if cfg.backbone.startswith('resnet'):
        from .resnet import ResNetWrapper
        backbone = ResNetWrapper(resnet=cfg.backbone, pretrained=cfg.pretrained, out_conv=False)
        return backbone
    elif cfg.backbone == 'convnextT':
        from .convnext import convnext_tiny
        backbone = convnext_tiny(pretrained=cfg.pretrained)
        return backbone
    elif cfg.backbone == 'dla34':
        from .dla34 import DLAWrapper
        backbone = DLAWrapper(pretrained=cfg.pretrained)
        return backbone
    elif cfg.backbone.startswith('pvtv2'):
        from .pvtv2 import get_pvtv2
        backbone = get_pvtv2(name=cfg.backbone, pretrained=cfg.pretrained)
        return backbone
    else:
        return None