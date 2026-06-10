import torch
import torch.nn as nn
import torchvision.models as models


def normal_form_to_points(
    cos_theta: torch.Tensor,
    sin_theta: torch.Tensor,
    rho: torch.Tensor,
    x_coords: torch.Tensor,
) -> torch.Tensor:
    """Convert Hesse normal form (cos·x + sin·y = ρ) to points at given x's.

    Shapes accepted:
      cos/sin/rho: [B] or [B, num_rows]
      x_coords:    [B, num_cols] or [B, num_rows, num_cols]
    """
    # For nearly horizontal lines (table rows) sin_theta is close to ±1, but a
    # near-zero sin_theta would blow up the division. Clamp magnitude, keep sign.
    sin_theta_safe = torch.clamp(sin_theta.abs(), min=1e-6) * torch.sign(sin_theta + 1e-8)

    if cos_theta.dim() == 1:
        y_coords = (rho.unsqueeze(1) - x_coords * cos_theta.unsqueeze(1)) / sin_theta_safe.unsqueeze(1)
        return torch.stack([x_coords, y_coords], dim=-1)

    # Per-row case: broadcast a single (B, num_cols) x grid across rows so every
    # row samples at the same column positions.
    if x_coords.dim() == 2:
        x_coords = x_coords.unsqueeze(1).expand(-1, cos_theta.size(1), -1)

    y_coords = (rho.unsqueeze(2) - x_coords * cos_theta.unsqueeze(2)) / sin_theta_safe.unsqueeze(2)
    return torch.stack([x_coords, y_coords], dim=-1)


def points_to_normal_form(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit a Hesse-form line to points via SVD. Differentiable end-to-end."""
    original_shape = points.shape
    if points.dim() == 4:
        B, num_rows, num_cols, _ = points.shape
        points = points.view(B * num_rows, num_cols, 2)
    else:
        B = points.size(0)
        num_rows = 1

    centroid = points.mean(dim=1, keepdim=True)
    centered = points - centroid

    # The line direction is the largest singular vector of the centered points;
    # the normal is the smallest, which is the second row of Vh in 2D.
    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
    normal = Vh[:, 1, :].clone()

    # Pin orientation so cos/sin are continuous across rows: flip if sin < 0.
    # Done multiplicatively because in-place writes on an SVD output would break
    # autograd through linalg.svd's backward.
    flip_sign = 1.0 - 2.0 * (normal[:, 1:2] < 0).float()
    normal = normal * flip_sign

    cos_theta = normal[:, 0]
    sin_theta = normal[:, 1]
    rho = (centroid.squeeze(1) * normal).sum(dim=1)

    if len(original_shape) == 4:
        cos_theta = cos_theta.view(B, num_rows)
        sin_theta = sin_theta.view(B, num_rows)
        rho = rho.view(B, num_rows)

    return cos_theta, sin_theta, rho


class NormalLinePredictor(nn.Module):
    def __init__(
        self,
        hidden_size: int = 256,
        num_rows: int = 18,
        num_cols: int = 5,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        use_pretrained: bool = True,
        predict_other_rows: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_lstm_layers = num_lstm_layers
        self.predict_other_rows = predict_other_rows

        backbone = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT if use_pretrained else None,
        )
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        backbone_out_channels = 512

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_proj = nn.Sequential(
            nn.Linear(backbone_out_channels, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # First head jointly predicts (cos, sin, rho) and the shared x-coords.
        # Tying x-coords across rows encodes the assumption that columns are
        # vertical — table rows lie on the same column grid.
        self.first_row_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3 + num_cols),
        )

        if predict_other_rows:
            # The LSTM only carries normal params row-to-row; the image features
            # are re-injected at every step to anchor the prediction visually.
            self.row_lstm = nn.LSTM(
                input_size=3 + hidden_size,
                hidden_size=hidden_size,
                num_layers=num_lstm_layers,
                dropout=dropout if num_lstm_layers > 1 else 0,
                batch_first=True,
            )

            self.row_output_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 3),
            )

    def _normalize_direction(
        self,
        cos_theta: torch.Tensor,
        sin_theta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The head outputs (cos, sin) without the unit-norm constraint, so we
        # project back onto the circle before plugging them into the Hesse form.
        norm = torch.sqrt(cos_theta ** 2 + sin_theta ** 2 + 1e-8)
        return cos_theta / norm, sin_theta / norm

    def forward(
        self,
        image: torch.Tensor,
        teacher_forcing_rows: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        B = image.size(0)
        device = image.device

        features = self.backbone(image)
        features = self.global_pool(features).flatten(1)
        features = self.feature_proj(features)

        first_row_out = self.first_row_head(features)

        first_cos = first_row_out[:, 0]
        first_sin = first_row_out[:, 1]
        first_rho = first_row_out[:, 2]
        # Tanh clamps x-coords to the normalised [-1, 1] image extent.
        x_coords = torch.tanh(first_row_out[:, 3:])

        first_cos, first_sin = self._normalize_direction(first_cos, first_sin)
        first_row_points = normal_form_to_points(first_cos, first_sin, first_rho, x_coords)

        if not self.predict_other_rows:
            return first_row_points.unsqueeze(1)

        all_rows = [first_row_points]
        h = torch.zeros(self.num_lstm_layers, B, self.hidden_size, device=device)
        c = torch.zeros(self.num_lstm_layers, B, self.hidden_size, device=device)

        current_cos = first_cos
        current_sin = first_sin
        current_rho = first_rho

        for row_idx in range(1, self.num_rows):
            # view(B, 1, 1) instead of unsqueeze(1): the LSTM input needs to be
            # 3D (B, T=1, F), and unsqueeze would only give 2D from a 1D scalar.
            lstm_input = torch.cat([
                current_cos.view(B, 1, 1),
                current_sin.view(B, 1, 1),
                current_rho.view(B, 1, 1),
                features.unsqueeze(1),
            ], dim=2)

            lstm_out, (h, c) = self.row_lstm(lstm_input, (h, c))
            lstm_out = lstm_out.squeeze(1)

            row_params = self.row_output_head(lstm_out)
            pred_cos, pred_sin = self._normalize_direction(row_params[:, 0], row_params[:, 1])
            pred_rho = row_params[:, 2]

            # Reuse the first row's x-coords so all rows share a column grid.
            row_points = normal_form_to_points(pred_cos, pred_sin, pred_rho, x_coords)
            all_rows.append(row_points)

            use_teacher = (
                teacher_forcing_rows is not None
                and torch.rand(1).item() < teacher_forcing_ratio
            )
            if use_teacher and teacher_forcing_rows is not None:
                # Feed back the GT normal params (re-fitted from GT points) so
                # the next LSTM step sees a clean, calibrated input.
                gt_row = teacher_forcing_rows[:, row_idx, :, :]
                gt_cos, gt_sin, gt_rho = points_to_normal_form(gt_row.unsqueeze(1))
                current_cos = gt_cos.squeeze(1)
                current_sin = gt_sin.squeeze(1)
                current_rho = gt_rho.squeeze(1)
            else:
                current_cos = pred_cos
                current_sin = pred_sin
                current_rho = pred_rho

        return torch.stack(all_rows, dim=1)
