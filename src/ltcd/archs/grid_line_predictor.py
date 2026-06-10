import torch
import torch.nn as nn
import torchvision.models as models


def normal_form_to_points(
    cos_theta: torch.Tensor,
    sin_theta: torch.Tensor,
    rho: torch.Tensor,
    sample_coords: torch.Tensor,
    is_horizontal: bool = True,
) -> torch.Tensor:
    """Sample points on a Hesse-form line at given coordinates.

    For horizontal lines we sample at x-coords and solve for y; for vertical
    lines we sample at y-coords and solve for x. The pivot lets us avoid the
    division-by-zero that slope-intercept form would hit at the orthogonal axis.
    """
    if is_horizontal:
        # Near-horizontal => sin_theta close to ±1; clamping magnitude and
        # preserving sign keeps the division stable without flipping rows.
        sin_safe = torch.clamp(sin_theta.abs(), min=1e-6) * torch.sign(sin_theta + 1e-8)
        if cos_theta.dim() == 1:
            y = (rho.unsqueeze(1) - sample_coords * cos_theta.unsqueeze(1)) / sin_safe.unsqueeze(1)
            return torch.stack([sample_coords, y], dim=-1)
        if sample_coords.dim() == 2:
            sample_coords = sample_coords.unsqueeze(1).expand(-1, cos_theta.size(1), -1)
        y = (rho.unsqueeze(2) - sample_coords * cos_theta.unsqueeze(2)) / sin_safe.unsqueeze(2)
        return torch.stack([sample_coords, y], dim=-1)

    cos_safe = torch.clamp(cos_theta.abs(), min=1e-6) * torch.sign(cos_theta + 1e-8)
    if sin_theta.dim() == 1:
        x = (rho.unsqueeze(1) - sample_coords * sin_theta.unsqueeze(1)) / cos_safe.unsqueeze(1)
        return torch.stack([x, sample_coords], dim=-1)
    if sample_coords.dim() == 2:
        sample_coords = sample_coords.unsqueeze(1).expand(-1, sin_theta.size(1), -1)
    x = (rho.unsqueeze(2) - sample_coords * sin_theta.unsqueeze(2)) / cos_safe.unsqueeze(2)
    return torch.stack([x, sample_coords], dim=-1)


def points_to_normal_form(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SVD-fit a Hesse-form line to points. Differentiable end-to-end."""
    original_shape = points.shape
    if points.dim() == 4:
        B, num_lines, num_points, _ = points.shape
        points = points.view(B * num_lines, num_points, 2)
    else:
        B = points.size(0)
        num_lines = 1

    centroid = points.mean(dim=1, keepdim=True)
    centered = points - centroid

    # Smallest singular vector ⇒ direction of least variance ⇒ line normal.
    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
    normal = Vh[:, 1, :].clone()

    # Multiplicative flip instead of indexed assignment: in-place writes on the
    # SVD output break autograd through linalg.svd.
    flip_sign = 1.0 - 2.0 * (normal[:, 1:2] < 0).float()
    normal = normal * flip_sign

    cos_theta = normal[:, 0]
    sin_theta = normal[:, 1]
    rho = (centroid.squeeze(1) * normal).sum(dim=1)

    if len(original_shape) == 4:
        cos_theta = cos_theta.view(B, num_lines)
        sin_theta = sin_theta.view(B, num_lines)
        rho = rho.view(B, num_lines)

    return cos_theta, sin_theta, rho


class UnifiedGridLinePredictor(nn.Module):
    """Single-LSTM grid predictor with a direction embedding.

    Both horizontal and vertical lines are produced by the same LSTM and head;
    a learnable embedding tells the network which orientation it should emit.
    Shared x/y coordinate heads ensure that intersection points are consistent
    across the two passes.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_rows: int = 18,
        num_cols: int = 5,
        # Kept for backwards compatibility with old checkpoints; the model now
        # samples num_cols points per horizontal line and num_rows per vertical.
        num_samples: int = 10,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        use_pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_samples = num_samples
        self.num_lstm_layers = num_lstm_layers

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

        # Shared sampling coordinates: horizontal lines all use the same x's,
        # vertical lines all use the same y's. This is what makes the grid
        # intersections consistent across the two directional passes.
        self.x_coords_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_cols),
            nn.Tanh(),
        )
        self.y_coords_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_rows),
            nn.Tanh(),
        )

        # 0 = horizontal, 1 = vertical. Encoded into a quarter of hidden_size
        # so it can't drown out the image features.
        self.direction_embedding = nn.Embedding(2, hidden_size // 4)
        self.combined_proj = nn.Sequential(
            nn.Linear(hidden_size + hidden_size // 4, hidden_size),
            nn.ReLU(),
        )

        self.first_line_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3),
        )

        self.lstm = nn.LSTM(
            input_size=3 + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 3),
        )

    def _normalize_direction(
        self,
        cos_t: torch.Tensor,
        sin_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Project (cos, sin) back onto the unit circle — the head has no
        # built-in norm constraint and the Hesse form requires it.
        norm = torch.sqrt(cos_t ** 2 + sin_t ** 2 + 1e-8)
        return cos_t / norm, sin_t / norm

    def _predict_lines(
        self,
        features: torch.Tensor,
        sample_coords: torch.Tensor,
        direction: int,
        num_lines: int,
    ) -> torch.Tensor:
        B = features.size(0)
        device = features.device

        dir_emb = self.direction_embedding(
            torch.tensor([direction], device=device),
        ).expand(B, -1)
        combined = self.combined_proj(torch.cat([features, dir_emb], dim=1))

        first_out = self.first_line_head(combined)
        first_cos, first_sin = self._normalize_direction(first_out[:, 0], first_out[:, 1])
        first_rho = first_out[:, 2]

        is_horizontal = (direction == 0)
        first_points = normal_form_to_points(
            first_cos, first_sin, first_rho, sample_coords, is_horizontal,
        )

        all_lines = [first_points]
        h = torch.zeros(self.num_lstm_layers, B, self.hidden_size, device=device)
        c = torch.zeros(self.num_lstm_layers, B, self.hidden_size, device=device)

        current_cos, current_sin, current_rho = first_cos, first_sin, first_rho

        for _ in range(1, num_lines):
            # view(B, 1, 1) instead of unsqueeze(1): the cat expects 3D inputs.
            lstm_input = torch.cat([
                current_cos.view(B, 1, 1),
                current_sin.view(B, 1, 1),
                current_rho.view(B, 1, 1),
                combined.unsqueeze(1),
            ], dim=2)

            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))
            params = self.output_head(lstm_out.squeeze(1))
            pred_cos, pred_sin = self._normalize_direction(params[:, 0], params[:, 1])
            pred_rho = params[:, 2]

            all_lines.append(
                normal_form_to_points(pred_cos, pred_sin, pred_rho, sample_coords, is_horizontal),
            )
            current_cos, current_sin, current_rho = pred_cos, pred_sin, pred_rho

        return torch.stack(all_lines, dim=1)

    def forward(
        self,
        image: torch.Tensor,
        predict_horizontal: bool = True,
        predict_vertical: bool = True,
    ) -> dict[str, torch.Tensor]:
        features = self.backbone(image)
        features = self.global_pool(features).flatten(1)
        features = self.feature_proj(features)

        x_coords = self.x_coords_head(features)
        y_coords = self.y_coords_head(features)

        result = {"x_coords": x_coords, "y_coords": y_coords}
        if predict_horizontal:
            result["horizontal"] = self._predict_lines(
                features, x_coords, direction=0, num_lines=self.num_rows,
            )
        if predict_vertical:
            result["vertical"] = self._predict_lines(
                features, y_coords, direction=1, num_lines=self.num_cols,
            )
        return result
