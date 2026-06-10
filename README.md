# ltcd — Lightweight Table Cell Detection

Master's thesis project.

**Topic:** Development of Lightweight Neural Architecture
for Table Cell Detection.

 **Supervisor:** Ivan Khodnenko.

The task is dense table grid detection on standardised document forms.

NOTE: This code was refactored using Claude and may still contain unresolved issues. For more information, please contact me via email glebmikloshevich@gmail.com

* The synthetic dataset itself is intentionally not committed; the generator
that produces it is.

## Install

```bash
uv sync

uv sync --group training
```

## Pipeline

1. **Generation.** `ltcd.generators.doc_generator.DocumentGenerator`.
2. **Train.** One script per architecture, paired with a config in
   `confs/`.
3. **Evaluate.** `scripts/measure_size.py` reports params, memory, and checkpoint size.
   the `inference_{model_name}` scripts compute PCK at 3 thresholdds, pixel error, and inference speed.

## Architectures


| Paradigm | Script | Architecture(s) |
|---|---|---|
| Segmentation | `train_segmentation_infinite.py` | TinyU-Net, UNeXt-S |
| Segmentation | `train_nafunet_heatmap.py` | NafUNet small / base, corners or all-points, WMSE or Dice+Focal losses|
| Cross-attention | `train_cross_attention_keypoint.py` | CrossAttentionKeypointDetector + Lightweight (Deformable) variant |
| Direct point | `train_first_row.py` | FirstRowRegressor |
| Direct point | `train_row_predictor.py` | LSTMRowPredictor |
| Direct point | `train_grid_line_predictor.py` | UnifiedGridLinePredictor (and in-repo YOLOKeypointPredictor via `model_type`) |
| Line prediction | `train_normal_line_predictor.py` | NormalLinePredictor (Hesse form) |

## Main results

```
LightweightCrossAttention   11.3M  PCK5 0.942  px 1.83   268 FPS
LSTMTablePredictor (refine)  2.3M  PCK5 0.917  px 1.98    92 FPS
NafUNet-Corners (WMSE)       1.9M  PCK5 0.901  px 24.7    39 FPS  (4 corners only)
```

On the mixed data (real + synthetic setup) **NafUNet-Corners (WMSE)** took the lead at PCK@5 = 0.838 
