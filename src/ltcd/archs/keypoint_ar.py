import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class KeypointAR(nn.Module):
    def __init__(self, hidden_size: int = 256, num_points: int = 75) -> None:
        super().__init__()

        backbone = models.resnet18(weights=None)
        layers = list(backbone.children())

        # Stop at layer2 (stride 8, 128 channels). Going deeper would shrink the
        # spatial map below the sampling granularity we need for grid_sample.
        self.encoder = nn.Sequential(*layers[:6])
        self.feat_dim = 128

        self.lstm = nn.LSTM(
            input_size=self.feat_dim + 2,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, 2)
        self.num_points = num_points

    def sample_feature(self, feat_map: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        # grid_sample wants a [B, H_out, W_out, 2] grid. We only sample one
        # point per item, so H_out = W_out = 1.
        grid = xy.unsqueeze(2)
        sampled = F.grid_sample(feat_map, grid, mode="bilinear", align_corners=True)
        return sampled.squeeze(-1).squeeze(-1).unsqueeze(1)

    def forward(
        self,
        image: torch.Tensor,
        start_xy: torch.Tensor,
        teacher_forcing_points: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feat_map = self.encoder(image)
        xy = start_xy
        outputs = []
        hidden = None

        for t in range(self.num_points):
            # 50% teacher forcing keeps the LSTM stable early in training when
            # its own predictions are far from the grid; halved instead of full
            # so the model still has to learn to recover from its own mistakes.
            if self.training and teacher_forcing_points is not None and torch.rand(1).item() < 0.5:
                xy = teacher_forcing_points[:, t:t + 1]

            local_feat = self.sample_feature(feat_map, xy)

            inp = torch.cat([local_feat, xy], dim=2)
            out, hidden = self.lstm(inp, hidden)

            # Unbounded output: targets are in [-1, 1] but no tanh is applied
            # here — the geometric/alignment losses are expected to pull the
            # predictions into range.
            xy = self.fc(out)

            outputs.append(xy)

        return torch.cat(outputs, dim=1)
