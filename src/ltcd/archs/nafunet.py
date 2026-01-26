# Based on "Simple Baselines for Image Restoration" paper and github code implementation
# https://arxiv.org/abs/2204.04676
# NAFUNet: U-Net architecture based on NAFNet blocks.
#


import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    """2D LayerNorm for channel-wise normalization using standard PyTorch LayerNorm."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W) -> (N, H, W, C) -> LayerNorm -> (N, C, H, W)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class SimpleGate(nn.Module):
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

        # Spatial processing branch
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, 1)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, 3, 1, 1, groups=dw_channels)
        self.sg = SimpleGate()
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, 1)

        # Channel processing branch (FFN)
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_expansion * channels, 1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_expansion * channels // 2, channels, 1)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connections."""
        shortcut = x

        # Spatial processing
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.conv3(x)
        x = self.drop_path(x)
        x = shortcut + x * self.beta

        # Channel processing (FFN)
        shortcut = x
        x = self.norm2(x)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        x = self.drop_path(x)
        x = shortcut + x * self.gamma

        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


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
    """U-Net architecture based on NAFNet blocks. """

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

        # increasing droprate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks))]

        self.input_conv = nn.Conv2d(in_channels, channels[0], 3, 1, 1)

        # Encoder
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

        # Decoder
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

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

        # encode
        skip_connections = []
        for level in range(self.num_levels):
            for block in self.encoders[level]: # TODO(Gleb): straighforward
                x = block(x)

            if level < self.num_levels - 1:
                skip_connections.append(x)
                x = self.downsamples[level](x)

        # decode
        for level in range(self.num_levels - 1):
            x = self.upsamples[level](x)

            # skip connection
            skip = skip_connections[-(level + 1)]

            # NOTE: can I do someting better ???
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)

            x = torch.cat([x, skip], dim=1)
            x = self.skip_convs[level](x)

            for block in self.decoders[level]:
                x = block(x)

        x = self.output(x)

        return x


class NAFUNetSmall(NAFUNet):
    """small NafUnet."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=24,
            num_blocks=[1, 1, 2, 4],
            drop_path_rate=0.05,
        )


class NAFUNetBase(NAFUNet):
    """Base NafUnet."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=32,
            num_blocks=[2, 2, 4, 8],
            drop_path_rate=0.1,
        )
