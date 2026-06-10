import torch
import torch.nn as nn
import torchvision.models as models


class FirstRowRegressor(nn.Module):
    """Proof-of-concept first-row predictor: CNN backbone + global pool + MLP.

    Used in Chapter 4 to confirm that a plain CNN-MLP can localise the 5-point
    first row well on the synthetic data, before scaling up to full-grid models.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_cols: int = 5,
        dropout: float = 0.1,
        use_pretrained: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_cols = num_cols

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None,
        )
        # Stop at layer2 (stride 8, 128 channels) — deep enough for semantic
        # context, shallow enough to keep the parameter budget at ~800K.
        self.encoder = nn.Sequential(*list(backbone.children())[:6])
        self.feat_dim = 128

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_cols * 2),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        B = image.size(0)
        feat_map = self.encoder(image)
        global_feat = self.global_pool(feat_map).view(B, -1)
        coords = self.head(global_feat).view(B, self.num_cols, 2)
        # Clamp to the normalised image extent so downstream sampling stays in
        # range — same convention as every other predictor in this module.
        return torch.tanh(coords)
