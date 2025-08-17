from .anchor_based_lane_detector import TwoStageAnchorBasedLaneDetector

def build_lane_model(cfg):
    model = TwoStageAnchorBasedLaneDetector(cfg=cfg)
    return model