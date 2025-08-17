def build_seg_head(cfg):
    from .segformer_head import get_segformer_head
    head = get_segformer_head(name=cfg.pvt, num_classes=cfg.num_seg_class)
    return head 