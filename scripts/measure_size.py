
from __future__ import annotations

import argparse
import json

from pathlib import Path

import torch

from torch import nn

from ltcd.archs.cross_attention_keypoint import LightweightCrossAttentionDetector
from ltcd.archs.deformable_grid_predictor import AffineGridPredictor
from ltcd.archs.keypoint_ar import KeypointAR
from ltcd.archs.lstm_table_predictor import LSTMTablePredictor
from ltcd.archs.nafunet import NAFUNetBase, NAFUNetSmall
from ltcd.archs.normal_line_predictor import NormalLinePredictor
from ltcd.archs.tinyunet import TinyUNet
from ltcd.archs.unext import UNextSmall


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def memory_mb(model: nn.Module) -> float:
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / (1024 * 1024)


def checkpoint_mb(path: Path | None) -> float:
    if path is None or not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def format_num(n: float | None) -> str:
    if n is None:
        return "N/A"
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return f"{n:.0f}"


MODEL_FACTORIES = {
    "NAFUNet-Small": lambda: NAFUNetSmall(in_channels=3, out_channels=1),
    "NAFUNet-Base": lambda: NAFUNetBase(in_channels=3, out_channels=1),
    "TinyU-Net": lambda: TinyUNet(in_channels=3, num_classes=1),
    "UNeXt-S": lambda: UNextSmall(num_classes=1, img_size=512),
    "NormalLinePredictor": lambda: NormalLinePredictor(num_rows=18, num_cols=5),
    "LSTMTablePredictor": lambda: LSTMTablePredictor(num_rows=18, num_cols=5),
    "KeypointAR": lambda: KeypointAR(num_points=90),
    "AffineGridPredictor": lambda: AffineGridPredictor(num_rows=18, num_cols=5),
    "LightCrossAttention": lambda: LightweightCrossAttentionDetector(num_rows=18, num_cols=5),
}


def measure(factory, checkpoint: Path | None) -> dict[str, float | int]:
    model = factory()
    model.eval()
    params = count_parameters(model)
    return {
        "params_total": params["total"],
        "params_trainable": params["trainable"],
        "memory_mb": memory_mb(model),
        "checkpoint_mb": checkpoint_mb(checkpoint),
    }


def print_table(results: dict[str, dict]) -> None:
    print(f"\n{'Model':<25} {'Params':>12} {'Trainable':>12} {'Mem (MB)':>10} {'Ckpt (MB)':>11}")
    print("-" * 75)
    for name in sorted(results, key=lambda n: results[n]["params_total"]):
        r = results[name]
        print(
            f"{name:<25} {format_num(r['params_total']):>12} "
            f"{format_num(r['params_trainable']):>12} "
            f"{r['memory_mb']:>10.2f} {r['checkpoint_mb']:>11.2f}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=None,
        help="Optional folder with <model>.pth files for ckpt-size measurement.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model_sizes.json"),
        help="Where to write the JSON report.",
    )
    args = parser.parse_args()

    results: dict[str, dict] = {}
    for name, factory in MODEL_FACTORIES.items():
        ckpt = args.checkpoints / f"{name}.pth" if args.checkpoints else None
        try:
            results[name] = measure(factory, ckpt)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {name}: {exc}")

    print_table(results)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
