import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class LSTMTablePredictor(nn.Module):
    """Row-by-row table predictor.

    Architecture: a regression head emits the first row directly from global
    features; the LSTM then unrolls the remaining rows, each step seeded by the
    previous row's coordinates plus features sampled at those points.

    With refine_first_row=True the LSTM also gets to refine the regression
    output before producing row 1 — this is the Chapter 4 winner on Setup B.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_rows: int = 18,
        num_cols: int = 5,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        use_pretrained: bool = False,
        refine_first_row: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_lstm_layers = num_lstm_layers
        self.refine_first_row = refine_first_row

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None,
        )
        # Stop at layer2: stride 8, 128 channels. Going deeper would shrink the
        # spatial map below the granularity needed for per-keypoint sampling.
        self.encoder = nn.Sequential(*list(backbone.children())[:6])
        self.feat_dim = 128

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.first_row_head = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_cols * 2),
        )

        # Input concatenates the previous row's coordinates and the features
        # sampled at those coordinates. The coordinates alone aren't enough
        # because the LSTM also needs to see what the image looks like there.
        self.lstm = nn.LSTM(
            input_size=num_cols * 2 + num_cols * self.feat_dim,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
        )

        self.row_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_cols * 2),
        )

    def sample_features_at_points(
        self,
        feat_map: torch.Tensor,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """Read num_cols feature vectors out of feat_map at the given points.

        points: [B, num_cols, 2] in [-1, 1]. Returns [B, num_cols * C] flattened
        so the LSTM input stays 1D per row.
        """
        B, num_cols, _ = points.shape

        # grid_sample expects [B, H_out, W_out, 2]; treat each keypoint as its
        # own H_out=1 row so we sample exactly num_cols features.
        sampled = F.grid_sample(
            feat_map,
            points.unsqueeze(2),
            mode="bilinear",
            align_corners=True,
            padding_mode="border",
        )
        return sampled.squeeze(-1).permute(0, 2, 1).reshape(B, -1)

    def forward(
        self,
        image: torch.Tensor,
        teacher_forcing_rows: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        B = image.size(0)

        feat_map = self.encoder(image)
        global_feat = self.global_pool(feat_map).view(B, -1)

        # tanh at the head: targets live in [-1, 1] and feeding an unbounded
        # value back into grid_sample on the next step would clamp out anything
        # that drifts past the image edge.
        first_row_init = torch.tanh(self.first_row_head(global_feat).view(B, self.num_cols, 2))

        all_rows = []
        hidden = None

        if self.refine_first_row:
            # Pass the regression estimate back through the LSTM to clean it
            # up. Errors in row 0 propagate to every later row, so this single
            # extra step has an outsized effect on final accuracy.
            prev_row = first_row_init
            lstm_input = torch.cat(
                [prev_row.view(B, -1), self.sample_features_at_points(feat_map, prev_row)],
                dim=1,
            ).unsqueeze(1)
            lstm_out, hidden = self.lstm(lstm_input, hidden)
            first_row = torch.tanh(self.row_head(lstm_out.squeeze(1)).view(B, self.num_cols, 2))
            all_rows.append(first_row)
            prev_row = first_row
        else:
            all_rows.append(first_row_init)
            prev_row = first_row_init

        for row_idx in range(1, self.num_rows):
            # Teacher forcing pulls prev_row back to GT with some probability so
            # the LSTM doesn't compound its own errors during training.
            if self.training and teacher_forcing_rows is not None and torch.rand(1).item() < teacher_forcing_ratio:
                prev_row = teacher_forcing_rows[:, row_idx - 1, :, :]

            lstm_input = torch.cat(
                [prev_row.view(B, -1), self.sample_features_at_points(feat_map, prev_row)],
                dim=1,
            ).unsqueeze(1)
            lstm_out, hidden = self.lstm(lstm_input, hidden)
            next_row = torch.tanh(self.row_head(lstm_out.squeeze(1)).view(B, self.num_cols, 2))

            all_rows.append(next_row)
            prev_row = next_row

        return torch.stack(all_rows, dim=1)
