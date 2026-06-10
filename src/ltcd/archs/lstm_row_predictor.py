import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class LSTMRowPredictor(nn.Module):
    """Per-point AR predictor inside a single row.

    Inner step of LSTMTablePredictor's row-by-row decoder, but predicting one
    column at a time. Useful as a proof-of-concept before extending to the
    whole table.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_cols: int = 5,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        use_pretrained: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_cols = num_cols
        self.num_lstm_layers = num_lstm_layers

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None,
        )
        # Stop at layer1 (stride 4, 64 channels). Per-point sampling needs a
        # finer spatial grid than the table-level version, since adjacent
        # columns can be only a few pixels apart at the working resolution.
        self.encoder = nn.Sequential(*list(backbone.children())[:5])
        self.feat_dim = 64

        self.lstm = nn.LSTM(
            input_size=self.feat_dim + 2,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2),
        )

    def sample_feature(self, feat_map: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        # grid_sample wants [B, H_out, W_out, 2]; one sample per item => 1x1.
        sampled = F.grid_sample(
            feat_map,
            xy.unsqueeze(2),
            mode="bilinear",
            align_corners=True,
            padding_mode="border",
        )
        return sampled.squeeze(-1).squeeze(-1).unsqueeze(1)

    def forward(
        self,
        image: torch.Tensor,
        start_xy: torch.Tensor,
        teacher_forcing_points: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        feat_map = self.encoder(image)
        xy = start_xy
        outputs = []
        hidden = None

        for t in range(self.num_cols):
            if self.training and teacher_forcing_points is not None and torch.rand(1).item() < teacher_forcing_ratio:
                xy = teacher_forcing_points[:, t:t + 1, :]

            local_feat = self.sample_feature(feat_map, xy)
            inp = torch.cat([local_feat, xy], dim=2)
            out, hidden = self.lstm(inp, hidden)
            xy = self.fc(out)
            # Without this clamp a single bad step can push the next sampling
            # location outside the image; grid_sample with padding_mode="border"
            # would then return constant edge features and collapse the row.
            xy = torch.clamp(xy, -1, 1)
            outputs.append(xy)

        return torch.cat(outputs, dim=1)
