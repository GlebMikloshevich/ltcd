import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class KeypointAR(nn.Module):
    """Autoregressive keypoint predictor."""
    def __init__(self, hidden_size: int = 256, num_points: int = 75) -> None:
        super().__init__()

        backbone = models.resnet18(weights=None)
        layers = list(backbone.children())

        self.encoder = nn.Sequential(
            layers[0],
            layers[1],
            layers[2],
            layers[3],
            layers[4],
        )
        self.feat_dim = 128

        # lstm decoder
        self.lstm = nn.LSTM(
            input_size=self.feat_dim + 2,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, 2)   # predicts next xy
        self.num_points = num_points

    def sample_feature(self, feat_map: torch.Tensor, xy: torch.Tensor):
        # [b, h out, w out, 2]
        grid = xy.unsqueeze(2)  # [B, 1, 1, 2]
        sampled = F.grid_sample(
            feat_map, grid,
            mode="bilinear",
            align_corners=True,
        )
        return sampled.squeeze(-1).squeeze(-1).unsqueeze(1)  # [b, 1, c]

    def forward(
            self,
            image: torch.Tensor,
            start_xy: torch.Tensor,
            teacher_forcing_points: torch.Tensor | None = None,
        ) -> torch.Tensor:
        feat_map = self.encoder(image)  # [b, c, h, w]
        xy = start_xy  # [b, 1, 2]
        outputs = []
        hidden = None

        for t in range(self.num_points):
            if (self.training and teacher_forcing_points is not None and torch.rand(1).item() < 0.5):
                xy = teacher_forcing_points[:, t:t+1]

            local_feat = self.sample_feature(feat_map, xy)  # [b, 1, c]

            inp = torch.cat([local_feat, xy], dim=2)  # [b, 1, c+2]
            out, hidden = self.lstm(inp, hidden)

            # predict point
            # NOTE: normalized
            xy = self.fc(out)  # [b, 1, 2]

            outputs.append(xy)

        return torch.cat(outputs, dim=1)


class KeypointARMultiScale(nn.Module):
    """KeypoinyAR with multisclae ResNet features."""
    def __init__(
        self,
        hidden_size: int = 256,
        num_points: int = 75,
        use_pretrained: bool = False,
    ) -> None:
        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None,
        )
        layers = list(backbone.children())

        self.layer0 = nn.Sequential(layers[0], layers[1], layers[2])
        self.layer1 = nn.Sequential(layers[3], layers[4])
        self.layer2 = layers[5]
        self.layer3 = layers[6]

        self.feat_dims = [64, 128, 256]

        self.fusion = nn.Sequential(
            nn.Conv2d(sum(self.feat_dims), 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.fused_dim = 128

        # lstm decoder
        self.lstm = nn.LSTM(
            input_size=self.fused_dim + 2,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 2),
        )

        self.num_points = num_points

    def extract_multiscale_features(self, image: torch.Tensor) -> torch.Tensor:
        # Extract features
        x = self.layer0(image)       # [b, 64, h/2, w/2]
        feat1 = self.layer1(x)       # [b, 64, h/4, w/4]
        feat2 = self.layer2(feat1)   # [b, 128, h/8, w/8]
        feat3 = self.layer3(feat2)   # [b, 256, h/16, w/16]

        # upsample size
        target_size = feat2.shape[2:]

        # TODO(Gleb): think about better way to merge them
        feat1_up = F.interpolate(feat1, size=target_size, mode="bilinear", align_corners=True)
        feat3_up = F.interpolate(feat3, size=target_size, mode="bilinear", align_corners=True)

        multi_scale_feat = torch.cat([feat1_up, feat2, feat3_up], dim=1)

        fused_feat = self.fusion(multi_scale_feat)

        return fused_feat

    def sample_feature(self, feat_map: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:

        grid = xy.unsqueeze(2)  # [b, 1, 1, 2]
        sampled = F.grid_sample(
            feat_map, grid,
            mode="bilinear",
            align_corners=True,
        )
        return sampled.squeeze(-1).squeeze(-1).unsqueeze(1)  # [b, 1, c]

    def forward(
        self,
        image: torch.Tensor,
        start_xy: torch.Tensor,
        teacher_forcing_points: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feat_map = self.extract_multiscale_features(image)  # [b, fused_dim, h, w]

        xy = start_xy  # [b, 1, 2]
        outputs = []
        hidden = None

        for t in range(self.num_points):
            # Teacher forcing
            if self.training and teacher_forcing_points is not None and torch.rand(1).item() < 0.5:
                xy = teacher_forcing_points[:, t:t+1]

            local_feat = self.sample_feature(feat_map, xy)  # [b, 1, fused_dim]

            inp = torch.cat([local_feat, xy], dim=2)  # [b, 1, fused_dim + 2]
            out, hidden = self.lstm(inp, hidden)

            # predict point
            # NOTE: normalized
            xy = self.fc(out)  # [b, 1, 2]

            outputs.append(xy)

        return torch.cat(outputs, dim=1)
