import torch
import torch.nn as nn
import torchvision.models as models


def create_template_grid(num_rows: int, num_cols: int) -> torch.Tensor:
    """Regular grid in [-1, 1] used as the structural prior for predictions."""
    y = torch.linspace(-1, 1, num_rows)
    x = torch.linspace(-1, 1, num_cols)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1)


class AffineGridPredictor(nn.Module):
    """Predicts a global affine transform of a template grid plus per-point offsets.

    The affine head captures the table's pose (translation, rotation, scale,
    shear); the offset head models local deformations the affine can't express.
    Decoupling the two regularises the offsets — they only carry small residual
    corrections, not the whole geometry.
    """

    def __init__(
        self,
        num_rows: int = 18,
        num_cols: int = 5,
        hidden_size: int = 256,
        offset_scale: float = 0.1,
        use_pretrained: bool = True,
        backbone_name: str = "resnet34",
    ) -> None:
        super().__init__()

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_points = num_rows * num_cols
        self.offset_scale = offset_scale

        # Buffer (not Parameter) so the template moves with the model to
        # device/dtype but doesn't get gradients.
        self.register_buffer("template", create_template_grid(num_rows, num_cols))

        if backbone_name == "resnet18":
            backbone = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if use_pretrained else None,
            )
            backbone_channels = 512
        elif backbone_name == "resnet34":
            backbone = models.resnet34(
                weights=models.ResNet34_Weights.DEFAULT if use_pretrained else None,
            )
            backbone_channels = 512
        else:
            backbone = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if use_pretrained else None,
            )
            backbone_channels = 2048

        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.feature_proj = nn.Sequential(
            nn.Linear(backbone_channels, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.affine_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 6),
        )

        self.offset_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_points * 2),
            # Tanh + offset_scale clip: bounds offsets to ±offset_scale so they
            # behave as small local corrections, never overpowering the affine.
            nn.Tanh(),
        )

        self._init_affine_head()

    def _init_affine_head(self) -> None:
        # Initialise the affine to identity (1 0 0 / 0 1 0). Without this the
        # first forward pass scatters predictions randomly across the image and
        # the offset head has no stable signal to learn from.
        nn.init.zeros_(self.affine_head[-1].weight)
        with torch.no_grad():
            self.affine_head[-1].bias.data = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        B = image.size(0)

        features = self.backbone(image)
        features = self.global_pool(features).flatten(1)
        features = self.feature_proj(features)

        affine_matrix = self.affine_head(features).view(B, 2, 3)

        # Homogeneous coordinates so the bias term in the affine is just a
        # third column multiplied by the appended 1's.
        template_flat = self.template.view(1, -1, 2).expand(B, -1, -1)
        ones = torch.ones(B, self.num_points, 1, device=image.device)
        template_homo = torch.cat([template_flat, ones], dim=-1)
        base_grid = torch.bmm(template_homo, affine_matrix.transpose(1, 2))
        base_grid = base_grid.view(B, self.num_rows, self.num_cols, 2)

        offsets = self.offset_head(features).view(B, self.num_rows, self.num_cols, 2)
        offsets = offsets * self.offset_scale

        # Clamp to the normalised image extent — same convention as every
        # other predictor, lets downstream consumers assume in-range outputs.
        grid = torch.clamp(base_grid + offsets, -1, 1)

        return {
            "grid": grid,
            "affine_params": affine_matrix,
            "offsets": offsets,
            "base_grid": base_grid,
        }
