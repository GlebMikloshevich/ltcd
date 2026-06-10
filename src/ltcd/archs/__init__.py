import importlib

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ltcd.archs.cross_attention_keypoint import (
        CrossAttentionKeypointDetector,
        LightweightCrossAttentionDetector,
    )
    from ltcd.archs.deformable_grid_predictor import AffineGridPredictor
    from ltcd.archs.first_row_regressor import FirstRowRegressor
    from ltcd.archs.fpn import FPN, FPNWithAttention, ResNetFPNBackbone
    from ltcd.archs.grid_line_predictor import UnifiedGridLinePredictor
    from ltcd.archs.keypoint_ar import KeypointAR
    from ltcd.archs.lstm_row_predictor import LSTMRowPredictor
    from ltcd.archs.lstm_table_predictor import LSTMTablePredictor
    from ltcd.archs.nafunet import NAFUNet, NAFUNetBase, NAFUNetSmall
    from ltcd.archs.normal_line_predictor import NormalLinePredictor
    from ltcd.archs.tinyunet import TinyUNet
    from ltcd.archs.unext import UNext, UNextSmall
    from ltcd.archs.yolo_keypoint_predictor import (
        YOLOKeypointPredictor,
        YOLOKeypointPredictorMedium,
        YOLOKeypointPredictorSmall,
    )

__all__ = [
    "AffineGridPredictor",
    "CrossAttentionKeypointDetector",
    "FPN",
    "FPNWithAttention",
    "FirstRowRegressor",
    "KeypointAR",
    "LSTMRowPredictor",
    "LSTMTablePredictor",
    "LightweightCrossAttentionDetector",
    "NAFUNet",
    "NAFUNetBase",
    "NAFUNetSmall",
    "NormalLinePredictor",
    "ResNetFPNBackbone",
    "TinyUNet",
    "UNext",
    "UNextSmall",
    "UnifiedGridLinePredictor",
    "YOLOKeypointPredictor",
    "YOLOKeypointPredictorMedium",
    "YOLOKeypointPredictorSmall",
]

# NOTE: lazy imports not to raise error during experiments with different model archs
_lazy_imports = {
    "AffineGridPredictor": "ltcd.archs.deformable_grid_predictor",
    "CrossAttentionKeypointDetector": "ltcd.archs.cross_attention_keypoint",
    "FPN": "ltcd.archs.fpn",
    "FPNWithAttention": "ltcd.archs.fpn",
    "FirstRowRegressor": "ltcd.archs.first_row_regressor",
    "KeypointAR": "ltcd.archs.keypoint_ar",
    "LSTMRowPredictor": "ltcd.archs.lstm_row_predictor",
    "LSTMTablePredictor": "ltcd.archs.lstm_table_predictor",
    "LightweightCrossAttentionDetector": "ltcd.archs.cross_attention_keypoint",
    "NAFUNet": "ltcd.archs.nafunet",
    "NAFUNetBase": "ltcd.archs.nafunet",
    "NAFUNetSmall": "ltcd.archs.nafunet",
    "NormalLinePredictor": "ltcd.archs.normal_line_predictor",
    "ResNetFPNBackbone": "ltcd.archs.fpn",
    "TinyUNet": "ltcd.archs.tinyunet",
    "UNext": "ltcd.archs.unext",
    "UNextSmall": "ltcd.archs.unext",
    "UnifiedGridLinePredictor": "ltcd.archs.grid_line_predictor",
    "YOLOKeypointPredictor": "ltcd.archs.yolo_keypoint_predictor",
    "YOLOKeypointPredictorMedium": "ltcd.archs.yolo_keypoint_predictor",
    "YOLOKeypointPredictorSmall": "ltcd.archs.yolo_keypoint_predictor",
}


def __getattr__(name: str):
    if name in _lazy_imports:
        module = importlib.import_module(_lazy_imports[name])
        return getattr(module, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
