# Adapted from the official TinyU-Net implementation:
# https://github.com/ChenJunren-Lab/TinyU-Net
# Formatting follows the upstream style; do not run ruff on this file.
# fmt: off

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):
    # Replicates the same-padding rule used in YOLOv5/Ultralytics so dilated
    # kernels still produce same-spatial output.
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.GELU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn   = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act  = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DWConv(Conv):
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        # gcd(c1, c2) groups makes this a depthwise conv when c1==c2 and a
        # grouped conv otherwise — single class covers both cases.
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class CMRF(nn.Module):
    """Cascade Multi-Receptive-Fields block: cheap multi-scale features through
    cascaded depthwise convs on disjoint channel slices instead of parallel
    branches. The cost is N depthwise 3×3 convs instead of N independent
    branches with their own pointwise projections."""

    def __init__(self, c1, c2, N=8, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.N       = N
        self.c       = int(c2 * e / self.N)
        self.add     = shortcut and c1 == c2

        self.pwconv1 = Conv(c1, c2 // self.N, 1, 1)
        self.pwconv2 = Conv(c2 // 2, c2, 1, 1)
        self.m       = nn.ModuleList(DWConv(self.c, self.c, k=3, act=False) for _ in range(N - 1))

    def forward(self, x):
        x_residual = x
        x = self.pwconv1(x)

        # Even/odd channel split: the even half is the residual, the odd half
        # is iterated through N-1 depthwise convs to grow the receptive field.
        x = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
        x.extend(m(x[-1]) for m in self.m)
        x[0] = x[0] + x[1]
        x.pop(1)

        y = torch.cat(x, dim=1)
        y = self.pwconv2(y)
        return x_residual + y if self.add else y


class UNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cmrf       = CMRF(in_channels, out_channels)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.cmrf(x)
        # Returns (downsampled, skip): the caller forwards the downsampled
        # tensor and routes the skip to the symmetric decoder block.
        return self.downsample(x), x


class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cmrf     = CMRF(in_channels, out_channels)
        self.upsample = F.interpolate

    def forward(self, x, skip_connection):
        # Bicubic upsample — bilinear loses thin-line detail at this scale.
        x = self.upsample(x, scale_factor=2, mode='bicubic', align_corners=False)
        x = torch.cat([x, skip_connection], dim=1)
        return self.cmrf(x)


class TinyUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        # in_filters account for the channel-concat with skip connections;
        # out_filters are the post-CMRF channel counts.
        in_filters  = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]

        self.encoder1 = UNetEncoder(in_channels, 64)
        self.encoder2 = UNetEncoder(64, 128)
        self.encoder3 = UNetEncoder(128, 256)
        self.encoder4 = UNetEncoder(256, 512)

        self.decoder4 = UNetDecoder(in_filters[3], out_filters[3])
        self.decoder3 = UNetDecoder(in_filters[2], out_filters[2])
        self.decoder2 = UNetDecoder(in_filters[1], out_filters[1])
        self.decoder1 = UNetDecoder(in_filters[0], out_filters[0])
        self.final_conv = nn.Conv2d(out_filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        x, skip1 = self.encoder1(x)
        x, skip2 = self.encoder2(x)
        x, skip3 = self.encoder3(x)
        x, skip4 = self.encoder4(x)

        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        return self.final_conv(x)
