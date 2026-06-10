# Based on "Simple Baselines for Image Restoration"
# https://arxiv.org/abs/2204.04676

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # nn.LayerNorm expects channels last; conv tensors are channels first.
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class SimpleGate(nn.Module):
    # Drop-in replacement for GELU/ReLU in NAFNet: half the channels gate the
    # other half multiplicatively. No exponentials, no softmax — cheaper and
    # the paper shows it performs on par with explicit activations.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dw_expansion: int = 2,
        ffn_expansion: int = 2,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()

        dw_channels = channels * dw_expansion

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, 1)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, 3, 1, 1, groups=dw_channels)
        self.sg = SimpleGate()
        # SimpleGate halves the channel count, so the projection back is from
        # dw_channels // 2, not dw_channels.
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, 1)

        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_expansion * channels, 1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_expansion * channels // 2, channels, 1)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Zero-init residual scales so the block starts as identity. They become
        # learnable, which lets the network ramp each branch in only if useful.
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.conv3(x)
        x = self.drop_path(x)
        x = shortcut + x * self.beta

        shortcut = x
        x = self.norm2(x)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        x = self.drop_path(x)
        x = shortcut + x * self.gamma

        return x


class DropPath(nn.Module):
    # Stochastic depth: drops whole residual branches per-sample, not per-pixel
    # like nn.Dropout. The branch is kept with probability (1 - drop_prob) and
    # the kept ones are scaled up by 1/(1 - drop_prob) so the expected output
    # matches inference.
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class NAFUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        num_blocks: list[int] | None = None,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()

        if num_blocks is None:
            num_blocks = [2, 2, 4, 8]

        self.num_levels = len(num_blocks)
        channels = [base_channels * (2**i) for i in range(self.num_levels)]

        # Linear drop-path schedule: deeper blocks drop harder. Standard recipe
        # from the timm reference implementation.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks))]

        self.input_conv = nn.Conv2d(in_channels, channels[0], 3, 1, 1)

        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        idx = 0
        for level in range(self.num_levels):
            blocks = nn.ModuleList()
            for _ in range(num_blocks[level]):
                blocks.append(
                    NAFBlock(
                        channels[level],
                        dw_expansion=2,
                        ffn_expansion=2,
                        drop_path=dpr[idx],
                    ),
                )
                idx += 1
            self.encoders.append(blocks)

            if level < self.num_levels - 1:
                self.downsamples.append(
                    DownsampleBlock(channels[level], channels[level + 1]),
                )

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        # Decoder blocks skip drop-path: depth is symmetric to the encoder, so
        # applying it again would double-count the regularisation.
        for level in range(self.num_levels - 1, 0, -1):
            self.upsamples.append(UpsampleBlock(channels[level], channels[level - 1]))
            self.skip_convs.append(
                nn.Conv2d(channels[level - 1] * 2, channels[level - 1], 1),
            )

            blocks = nn.ModuleList()
            for _ in range(num_blocks[level - 1]):
                blocks.append(
                    NAFBlock(
                        channels[level - 1],
                        dw_expansion=2,
                        ffn_expansion=2,
                        drop_path=0.0,
                    ),
                )
            self.decoders.append(blocks)

        self.output = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], 3, 1, 1),
            LayerNorm2d(channels[0]),
            nn.Conv2d(channels[0], out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)

        skip_connections = []
        for level in range(self.num_levels):
            for block in self.encoders[level]:
                x = block(x)

            if level < self.num_levels - 1:
                skip_connections.append(x)
                x = self.downsamples[level](x)

        for level in range(self.num_levels - 1):
            x = self.upsamples[level](x)

            skip = skip_connections[-(level + 1)]

            # ConvTranspose2d with stride 2 only matches the encoder shape when
            # the input H/W is divisible by 2**(num_levels-1). For arbitrary
            # input sizes (e.g. odd values from the augmentation pipeline) we
            # need to nudge the upsampled tensor back onto the skip's grid.
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)

            x = torch.cat([x, skip], dim=1)
            x = self.skip_convs[level](x)

            for block in self.decoders[level]:
                x = block(x)

        return self.output(x)


class NAFUNetSmall(NAFUNet):
    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=24,
            num_blocks=[1, 1, 2, 4],
            drop_path_rate=0.05,
        )


class NAFUNetBase(NAFUNet):
    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=32,
            num_blocks=[2, 2, 4, 8],
            drop_path_rate=0.1,
        )
