import math
from pyexpat import model
import torch
from torch import nn
from CBAM import CBAM
from RSBU import RSBU_CW

class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)
        self.CBAM = CBAM(channel=64)
    def forward(self, x):
        block1 = self.block1(x)

        y = RSBU_CW(in_channels=64, out_channels=64)(block1)
        block2 = self.CBAM(y)

        y = RSBU_CW(in_channels=64, out_channels=64)(block2)
        block3 = self.CBAM(y)

        y = RSBU_CW(in_channels=64, out_channels=64)(block3)
        block4 = self.CBAM(y)

        y = RSBU_CW(in_channels=64, out_channels=64)(block4)
        block5 = self.CBAM(y)

        y = RSBU_CW(in_channels=64, out_channels=64)(block5)
        block6 = self.CBAM(y)

        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        block_11 = self.block1(x)

        y = RSBU_CW(in_channels=64, out_channels=64)(block_11)
        block_12 = self.CBAM(y)

        y = RSBU_CW(in_channels=64, out_channels=64)(block_12)
        block_13 = self.CBAM(y)

        y = RSBU_CW(in_channels=64, out_channels=64)(block_13)
        block_14 = self.CBAM(y)

        y = RSBU_CW(in_channels=64, out_channels=64)(block_14)
        block_15 = self.CBAM(y)

        y = RSBU_CW(in_channels=64, out_channels=64)(block_15)
        block_16 = self.CBAM(y)

        block_17 = self.block7(block_16)
        block_18 = self.block8(block_11 + block_17)

        feature = torch.add(block8, block_18)
        feature = (torch.tanh(feature) + 1) / 2

        return feature

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

