import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PositionalEncoding2D(nn.Module):
    def __init__(self, hidden_size: int, temperature: float = 10000.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, C, H, W = x.shape
        device = x.device

        # Coordinates in [0, 1] so the encoding is resolution-independent — the
        # backbone may emit different H/W depending on input image size.
        y_pos = torch.arange(H, device=device).float() / (H - 1 + 1e-6)
        x_pos = torch.arange(W, device=device).float() / (W - 1 + 1e-6)
        y_grid, x_grid = torch.meshgrid(y_pos, x_pos, indexing="ij")

        # Half the channels encode x, half encode y; within each half we use the
        # standard sinusoidal scheme (sin/cos pairs at C/4 geometric frequencies).
        dim_half = C // 4
        dim_t = torch.arange(dim_half, device=device).float()
        dim_t = self.temperature ** (2 * dim_t / dim_half)

        pos_x = x_grid.unsqueeze(-1) / dim_t
        pos_y = y_grid.unsqueeze(-1) / dim_t
        pos_x = torch.stack([pos_x.sin(), pos_x.cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y.sin(), pos_y.cos()], dim=-1).flatten(-2)

        pos = torch.cat([pos_y, pos_x], dim=-1).permute(2, 0, 1).unsqueeze(0)
        return x + pos


class CrossAttentionKeypointDetector(nn.Module):
    def __init__(
        self,
        num_rows: int = 18,
        num_cols: int = 5,
        hidden_size: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 4,
        dropout: float = 0.1,
        use_pretrained: bool = True,
        backbone_name: str = "resnet34",
    ) -> None:
        super().__init__()

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_keypoints = num_rows * num_cols
        self.hidden_size = hidden_size

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
        self.input_proj = nn.Conv2d(backbone_channels, hidden_size, 1)
        self.pos_encoding = PositionalEncoding2D(hidden_size)

        # Small std so the initial queries don't dominate the position embedding
        # and the decoder can learn from a near-neutral starting point.
        self.keypoint_queries = nn.Parameter(
            torch.randn(self.num_keypoints, hidden_size) * 0.02,
        )

        # Row and column are encoded separately and concatenated; this makes the
        # structural prior explicit instead of forcing the network to discover
        # it from a flat positional embedding.
        self.row_embed = nn.Embedding(num_rows, hidden_size // 2)
        self.col_embed = nn.Embedding(num_cols, hidden_size // 2)
        self.query_pos_proj = nn.Linear(hidden_size, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.coord_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 2),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.row_embed.weight, std=0.02)
        nn.init.normal_(self.col_embed.weight, std=0.02)

    def _get_query_positions(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # Column-major: keypoint i corresponds to (row = i // num_cols,
        # col = i %  num_cols), matching the dataset annotation order.
        row_idx = torch.arange(self.num_rows, device=device)
        col_idx = torch.arange(self.num_cols, device=device)
        row_grid, col_grid = torch.meshgrid(row_idx, col_idx, indexing="ij")

        row_emb = self.row_embed(row_grid.flatten())
        col_emb = self.col_embed(col_grid.flatten())
        pos_emb = self.query_pos_proj(torch.cat([row_emb, col_emb], dim=-1))
        return pos_emb.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        B = image.size(0)
        device = image.device

        features = self.backbone(image)
        features = self.input_proj(features)
        features = self.pos_encoding(features)

        # Flatten H'×W' into a token sequence so the transformer can attend
        # globally regardless of the backbone's stride.
        features_flat = features.flatten(2).permute(0, 2, 1)
        memory = self.encoder(features_flat)

        queries = self.keypoint_queries.unsqueeze(0).expand(B, -1, -1)
        queries = queries + self._get_query_positions(B, device)

        decoded = self.decoder(queries, memory)
        coords = self.coord_head(decoded)
        grid = coords.view(B, self.num_rows, self.num_cols, 2)

        return {"grid": grid, "keypoints_flat": coords}


class LightweightCrossAttentionDetector(nn.Module):
    def __init__(
        self,
        num_rows: int = 18,
        num_cols: int = 5,
        hidden_size: int = 128,
        num_sample_points: int = 4,
        dropout: float = 0.1,
        use_pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_keypoints = num_rows * num_cols
        self.hidden_size = hidden_size
        self.num_sample_points = num_sample_points

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if use_pretrained else None,
        )
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.input_proj = nn.Conv2d(512, hidden_size, 1)

        # Reference grid laid out in [-0.8, 0.8] (not the full [-1, 1] extent)
        # so the initial sampling positions sit safely inside the image padding
        # and don't collapse to the corners at the first forward pass.
        row_pos = torch.linspace(-0.8, 0.8, num_rows)
        col_pos = torch.linspace(-0.8, 0.8, num_cols)
        grid_y, grid_x = torch.meshgrid(row_pos, col_pos, indexing="ij")
        ref_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        self.reference_points = nn.Parameter(ref_points)

        self.offset_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_sample_points * 2),
        )
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_sample_points),
        )
        self.keypoint_embed = nn.Parameter(
            torch.randn(self.num_keypoints, hidden_size) * 0.02,
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
            nn.Tanh(),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        B = image.size(0)

        features = self.input_proj(self.backbone(image))
        ref_points = self.reference_points.unsqueeze(0).expand(B, -1, -1)

        # grid_sample expects a [B, H_out, W_out, 2] grid. Treat each of the N
        # keypoints as an H_out=N row of W_out=1 samples to read off N features.
        init_features = F.grid_sample(
            features,
            ref_points.unsqueeze(2),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(-1).permute(0, 2, 1)

        keypoint_emb = self.keypoint_embed.unsqueeze(0).expand(B, -1, -1)
        combined = init_features + keypoint_emb

        # 0.1 scale keeps offsets small early in training so the deformable
        # sampling stays near the structural prior of the reference grid.
        offsets = self.offset_net(combined).view(
            B, self.num_keypoints, self.num_sample_points, 2,
        ) * 0.1

        sample_points = ref_points.unsqueeze(2) + offsets
        # Clamp to the grid_sample valid range; without this the gradient could
        # vanish on samples that drift outside [-1, 1] (padding_mode="border"
        # returns the same edge value for any out-of-range coordinate).
        sample_points = torch.clamp(sample_points, -1, 1)

        sampled = F.grid_sample(
            features,
            sample_points.view(B, -1, 1, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(-1).permute(0, 2, 1)
        sampled = sampled.view(B, self.num_keypoints, self.num_sample_points, -1)

        attn_weights = F.softmax(self.attention_net(combined), dim=-1)
        aggregated = (sampled * attn_weights.unsqueeze(-1)).sum(dim=2)

        coords = self.output_head(torch.cat([combined, aggregated], dim=-1))
        grid = coords.view(B, self.num_rows, self.num_cols, 2)

        return {
            "grid": grid,
            "keypoints_flat": coords,
            "reference_points": ref_points,
            "sample_offsets": offsets,
        }
