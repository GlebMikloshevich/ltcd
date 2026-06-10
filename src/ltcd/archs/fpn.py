import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """Feature Pyramid Network with optional P6/P7 extra blocks.

    Lateral 1×1 convs match channels; top-down nearest upsampling merges
    higher-level semantics into lower-level localisation features; the
    3×3 output convs remove the aliasing introduced by nearest-neighbour
    upsampling — without them the lateral seams stay visible in attention maps.
    """

    def __init__(
        self,
        in_channels_list: list[int],
        out_channels: int = 256,
        extra_blocks: bool = False,
    ) -> None:
        super().__init__()

        self.out_channels = out_channels
        self.extra_blocks = extra_blocks

        self.lateral_convs = nn.ModuleList(
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels_list
        )
        self.output_convs = nn.ModuleList(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        )

        if extra_blocks:
            self.extra_conv1 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
            self.extra_conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        # features: low → high resolution. We rebuild in the reverse direction
        # so each level absorbs the higher-level (semantically richer) signal.
        laterals = [conv(f) for f, conv in zip(features, self.lateral_convs)]

        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], mode="nearest")
            laterals[i - 1] = laterals[i - 1] + upsampled

        outputs = [conv(lat) for lat, conv in zip(laterals, self.output_convs)]

        if self.extra_blocks:
            outputs.append(self.extra_conv1(outputs[-1]))
            # ReLU on P7 only — matches RetinaNet's original recipe and keeps
            # the deepest level positive-only.
            outputs.append(F.relu(self.extra_conv2(outputs[-1])))

        return outputs


class FPNWithAttention(nn.Module):
    def __init__(
        self,
        in_channels_list: list[int],
        out_channels: int = 256,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.fpn = FPN(in_channels_list, out_channels)
        self.attention = nn.ModuleList(
            ChannelAttention(out_channels, reduction) for _ in in_channels_list
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        fpn_features = self.fpn(features)
        return [attn(feat) for feat, attn in zip(fpn_features, self.attention)]


class ChannelAttention(nn.Module):
    # CBAM-style: avg-pool and max-pool produce complementary statistics. Each
    # goes through a shared MLP; their sum is sigmoid-gated to a per-channel
    # multiplicative mask.
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        avg_out = self.fc(self.avg_pool(x).view(B, C))
        max_out = self.fc(self.max_pool(x).view(B, C))
        attn = self.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        return x * attn


class ResNetFPNBackbone(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet34",
        out_channels: int = 256,
        pretrained: bool = True,
        freeze_bn: bool = False,
        use_attention: bool = False,
    ) -> None:
        super().__init__()

        import torchvision.models as models

        if backbone_name == "resnet18":
            backbone = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None,
            )
            in_channels_list = [64, 128, 256, 512]
        elif backbone_name == "resnet34":
            backbone = models.resnet34(
                weights=models.ResNet34_Weights.DEFAULT if pretrained else None,
            )
            in_channels_list = [64, 128, 256, 512]
        elif backbone_name == "resnet50":
            backbone = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None,
            )
            # ResNet50 uses bottleneck blocks: 4× wider on every stage.
            in_channels_list = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
        )
        self.layer1 = backbone.layer1  # C2: stride 4
        self.layer2 = backbone.layer2  # C3: stride 8
        self.layer3 = backbone.layer3  # C4: stride 16
        self.layer4 = backbone.layer4  # C5: stride 32

        if use_attention:
            self.fpn = FPNWithAttention(in_channels_list, out_channels)
        else:
            self.fpn = FPN(in_channels_list, out_channels)

        self.out_channels = out_channels

        if freeze_bn:
            self._freeze_bn()

    def _freeze_bn(self) -> None:
        # Mostly used when fine-tuning from ImageNet weights with tiny batches
        # — running stats from a small batch are noisy and can degrade
        # convergence relative to the pretrained statistics.
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return self.fpn([c2, c3, c4, c5])
