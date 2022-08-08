import math
import torch
from torch import nn
from CBAM import CBAM
from RSBU import RSBU_CW

class Generator(nn.Module):
    def __init__(self, scale_factor):
        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.rfb = ReceptiveFieldBlock(64,64)
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)

        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)

        block_add = torch.add(block1 + block7,block1 + block7)
        
        block8 = self.rfb(block_add)

        block9 = self.block8(block8)

        return (torch.tanh(block9) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.RSBU_CW = RSBU_CW(channels, channels)

    def forward(self, x):
        residual = self.RSBU_CW(x)

        return x + residual

# class ResidualBlock(nn.Module):
#     def __init__(self, channels):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(channels)
#         self.prelu = nn.PReLU()
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(channels)

#     def forward(self, x):
#         residual = self.conv1(x)
#         residual = self.bn1(residual)
#         residual = self.prelu(residual)
#         residual = self.conv2(residual)
#         residual = self.bn2(residual)

#         return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

class ReceptiveFieldBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """Modules introduced in RFBNet paper.
        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
        """

        super(ReceptiveFieldBlock, self).__init__()
        branch_channels = in_channels // 4

        # shortcut layer
        self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0))

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, (1, 1), (1, 1), (0, 0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(branch_channels, branch_channels, (3, 3), (1, 1), (1, 1), dilation=1),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, (1, 1), (1, 1), (0, 0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(branch_channels, branch_channels, (1, 3), (1, 1), (0, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(branch_channels, branch_channels, (3, 3), (1, 1), (3, 3), dilation=3),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, (1, 1), (1, 1), (0, 0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(branch_channels, branch_channels, (3, 1), (1, 1), (1, 0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(branch_channels, branch_channels, (3, 3), (1, 1), (3, 3), dilation=3),
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels // 2, (1, 1), (1, 1), (0, 0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(branch_channels // 2, (branch_channels // 4) * 3, (1, 3), (1, 1), (0, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d((branch_channels // 4) * 3, branch_channels, (3, 1), (1, 1), (1, 0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(branch_channels, branch_channels, (3, 3), (1, 1), (5, 5), dilation=5),
        )

        self.conv_linear = nn.Conv2d(4 * branch_channels, out_channels, (1, 1), (1, 1), (0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)
        shortcut = torch.mul(shortcut, 0.2)

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        out = torch.cat([branch1, branch2, branch3, branch4], 1)
        out = self.conv_linear(out)
        out = torch.add(out, shortcut)

        return out