
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from ltcd.archs.fpn import ResNetFPNBackbone


def assign_keypoints_to_grid(
    gt_points: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Assign GT keypoints to the nearest cell in a grid_h × grid_w detection grid.

    Each keypoint maps to exactly one cell (nearest centroid). Cells with no assigned
    keypoint become negatives (conf_target = 0). Two keypoints colliding in the same
    cell is unlikely at 64×64 for typical tables, but increase grid size if needed.

    Args:
        gt_points: [B, N, 2]  normalized coordinates in [-1, 1]
        grid_h:    detection grid height (= feat_h = image_h // 8)
        grid_w:    detection grid width  (= feat_w = image_w // 8)

    Returns:
        coord_targets: [B, grid_h, grid_w, 2]  — GT coords at positive cells
        conf_targets:  [B, grid_h, grid_w]     — 1 at assigned cells, 0 elsewhere
    """
    B, N, _ = gt_points.shape
    device = gt_points.device

    xs = torch.linspace(-1.0, 1.0, grid_w, device=device)
    ys = torch.linspace(-1.0, 1.0, grid_h, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    centers = torch.stack([xx, yy], dim=-1).view(1, 1, -1, 2)  # [1, 1, H*W, 2]

    dists = ((gt_points.unsqueeze(2) - centers) ** 2).sum(-1)  # [B, N, H*W]
    cell_idx = dists.argmin(dim=2)  # [B, N]

    coord_flat = torch.zeros(B, grid_h * grid_w, 2, device=device)
    conf_flat = torch.zeros(B, grid_h * grid_w, device=device)

    coord_flat.scatter_(1, cell_idx.unsqueeze(-1).expand(-1, -1, 2), gt_points)
    conf_flat.scatter_(1, cell_idx, torch.ones(B, N, device=device))

    return (
        coord_flat.view(B, grid_h, grid_w, 2),
        conf_flat.view(B, grid_h, grid_w),
    )


class YOLOKeypointLoss(nn.Module):

    def __init__(
        self,
        coord_weight: float = 5.0,
        conf_weight: float = 1.0,
        pos_weight: float = 10.0,
    ) -> None:
        super().__init__()
        self.coord_weight = coord_weight
        self.conf_weight = conf_weight
        self.register_buffer("bce_pos_weight", torch.tensor(pos_weight))

    def forward(
        self,
        pred_grid: torch.Tensor,        # [B, R, C, 2]   tanh'd coords
        pred_conf_logit: torch.Tensor,  # [B, R, C]       raw logits
        coord_targets: torch.Tensor,    # [B, R, C, 2]
        conf_targets: torch.Tensor,     # [B, R, C]       binary
    ) -> dict[str, torch.Tensor]:

        conf_loss = F.binary_cross_entropy_with_logits(
            pred_conf_logit, conf_targets,
            pos_weight=self.bce_pos_weight,
        )

        pos_mask = conf_targets > 0.5  # [B, R, C]
        if pos_mask.any():
            coord_loss = F.smooth_l1_loss(
                pred_grid[pos_mask],
                coord_targets[pos_mask],
                beta=0.1,
            )
        else:
            coord_loss = pred_grid.sum() * 0.0

        loss = self.conf_weight * conf_loss + self.coord_weight * coord_loss
        return {"loss": loss, "conf_loss": conf_loss, "coord_loss": coord_loss}


class YOLOKeypointHead(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, 1),  # → (x, y, conf_logit)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class YOLOKeypointPredictor(nn.Module):
    """
    Anchor-free YOLO-style grid keypoint predictor with confidence scores.

    Runs on the full H/8 × W/8 feature map — no adaptive pool. For a 512px input
    that is 64×64 = 4096 cells. GT keypoints are assigned to their nearest cell;
    unassigned cells are negatives. At inference, threshold `confidence` to obtain
    a variable number of keypoints without specifying any grid size.

    Output keys:
        "grid":        [B, H/8, W/8, 2]   tanh'd absolute coords
        "confidence":  [B, H/8, W/8]      sigmoid scores
        "conf_logit":  [B, H/8, W/8]      raw logits (for loss)
        "horizontal":  same as "grid"     (GridLineLoss compat)
        "vertical":    grid.permute(0,2,1,3)
    """

    def __init__(
        self,
        backbone_name: str = "resnet34",
        hidden_size: int = 256,
        use_pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.backbone = ResNetFPNBackbone(
            backbone_name=backbone_name,
            out_channels=hidden_size,
            pretrained=use_pretrained,
        )

        # Fuse P3 + P4↑ + P5↑ → [B, C, H/8, W/8]
        self.scale_fusion = nn.Sequential(
            nn.Conv2d(hidden_size * 3, hidden_size, 1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
        )

        self.head = YOLOKeypointHead(hidden_size)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        fpn = self.backbone(image)
        p3, p4, p5 = fpn[1], fpn[2], fpn[3]

        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode="nearest")
        p5_up = F.interpolate(p5, size=p3.shape[-2:], mode="nearest")
        fused = self.scale_fusion(torch.cat([p3, p4_up, p5_up], dim=1))

        raw = self.head(fused)  # [B, 3, H/8, W/8]

        xy = torch.tanh(raw[:, :2]).permute(0, 2, 3, 1)  # [B, R, C, 2]
        conf_logit = raw[:, 2]                             # [B, R, C]
        confidence = torch.sigmoid(conf_logit)             # [B, R, C]

        return {
            "grid":       xy,
            "confidence": confidence,
            "conf_logit": conf_logit,
            "horizontal": xy,
            "vertical":   xy.permute(0, 2, 1, 3).contiguous(),
        }

    @torch.no_grad()
    def decode(
        self,
        image: torch.Tensor,
        conf_threshold: float = 0.5,
    ) -> list[torch.Tensor]:
        """
        Inference helper. Returns a variable-length list of keypoints per image.

        Returns:
            List of length B, each element [K, 2] where K varies per image.
        """
        out = self.forward(image)
        grid = out["grid"]          # [B, R, C, 2]
        conf = out["confidence"]    # [B, R, C]

        results = []
        for b in range(image.size(0)):
            mask = conf[b] > conf_threshold       # [R, C]
            results.append(grid[b][mask])         # [K, 2]
        return results


class YOLOKeypointPredictorSmall(nn.Module):

    def __init__(
        self,
        hidden_size: int = 128,
        use_pretrained: bool = True,
    ) -> None:
        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if use_pretrained else None,
        )
        # children: [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc]
        # Keep through layer2 → stride 8, 128 channels.
        self.backbone = nn.Sequential(*list(backbone.children())[:6])

        if hidden_size == 128:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.Conv2d(128, hidden_size, 1, bias=False),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU(inplace=True),
            )

        self.head = YOLOKeypointHead(hidden_size)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.proj(self.backbone(image))
        raw = self.head(feat)  # [B, 3, H/8, W/8]

        xy = torch.tanh(raw[:, :2]).permute(0, 2, 3, 1)  # [B, R, C, 2]
        conf_logit = raw[:, 2]
        confidence = torch.sigmoid(conf_logit)

        return {
            "grid":       xy,
            "confidence": confidence,
            "conf_logit": conf_logit,
            "horizontal": xy,
            "vertical":   xy.permute(0, 2, 1, 3).contiguous(),
        }

    @torch.no_grad()
    def decode(
        self,
        image: torch.Tensor,
        conf_threshold: float = 0.5,
    ) -> list[torch.Tensor]:
        out = self.forward(image)
        grid = out["grid"]
        conf = out["confidence"]
        results = []
        for b in range(image.size(0)):
            mask = conf[b] > conf_threshold
            results.append(grid[b][mask])
        return results


class YOLOKeypointPredictorMedium(nn.Module):

    def __init__(
        self,
        hidden_size: int = 128,
        use_pretrained: bool = True,
    ) -> None:
        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if use_pretrained else None,
        )
        children = list(backbone.children())
        # children[0..4]: conv1, bn1, relu, maxpool, layer1 (stride 4, 64 ch)
        # children[5]: layer2 (stride 8, 128 ch)
        # children[6]: layer3 (stride 16, 256 ch)
        self.stem = nn.Sequential(*children[:5])
        self.layer2 = children[5]
        self.layer3 = children[6]

        self.lateral2 = nn.Conv2d(128, hidden_size, 1, bias=False)
        self.lateral3 = nn.Conv2d(256, hidden_size, 1, bias=False)
        self.smooth = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
        )

        self.head = YOLOKeypointHead(hidden_size)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(image)        # [B, 64, H/4, W/4]
        c2 = self.layer2(x)         # [B, 128, H/8, W/8]
        c3 = self.layer3(c2)        # [B, 256, H/16, W/16]

        p2 = self.lateral2(c2)
        p3 = self.lateral3(c3)
        p3_up = F.interpolate(p3, size=p2.shape[-2:], mode="nearest")
        feat = self.smooth(p2 + p3_up)

        raw = self.head(feat)  # [B, 3, H/8, W/8]

        xy = torch.tanh(raw[:, :2]).permute(0, 2, 3, 1)
        conf_logit = raw[:, 2]
        confidence = torch.sigmoid(conf_logit)

        return {
            "grid":       xy,
            "confidence": confidence,
            "conf_logit": conf_logit,
            "horizontal": xy,
            "vertical":   xy.permute(0, 2, 1, 3).contiguous(),
        }

    @torch.no_grad()
    def decode(
        self,
        image: torch.Tensor,
        conf_threshold: float = 0.5,
    ) -> list[torch.Tensor]:
        out = self.forward(image)
        grid = out["grid"]
        conf = out["confidence"]
        results = []
        for b in range(image.size(0)):
            mask = conf[b] > conf_threshold
            results.append(grid[b][mask])
        return results
