"""ResNet implementation taken from kuangliu on github
link: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch.nn as nn
import torch.nn.functional as F
import torch


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        x_skip = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x_skip = self.shortcut(x_skip)
        x = self.relu(x + x_skip)
        return x


class ResNet18(nn.Module):
    def __init__(self, num_classes=200):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_block(out_channels=64, stride=1)
        self.layer2 = self.make_block(out_channels=128, stride=2)
        self.layer3 = self.make_block(out_channels=256, stride=2)
        self.layer4 = self.make_block(out_channels=512, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def make_block(self, out_channels, stride):
        layers = []
        for stride in [stride, 1]:
            layers.append(ResNetBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x


class BottleneckBlock(nn.Module):
    """
    Bottleneck residual block used by ResNet50/101/152.

    Structure:
    - 1x1 conv: in_channels -> mid_channels (reduce)
    - batchnorm, ReLU
    - 3x3 conv: mid_channels -> mid_channels, stride=stride, padding=1
    - batchnorm, ReLU
    - 1x1 conv: mid_channels -> mid_channels * expansion (expand)
    - batchnorm
    - add shortcut, then ReLU

    The shortcut is a 1x1 conv + batchnorm when the spatial size or channel
    count changes, otherwise an identity (no-op).
    """
    expansion = 4

    def __init__(self, in_channels, mid_channels, stride=1):
        super(BottleneckBlock, self).__init__()
        out_channels = mid_channels * self.expansion

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        x_skip = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x_skip = self.shortcut(x_skip)
        x = self.relu(x + x_skip)
        return x


class ResNet50(nn.Module):
    def __init__(self, num_classes=200):
        super(ResNet50, self).__init__()
        self.in_channels = 64
        # Same 3x3 stride-1 stem as ResNet18 above (CIFAR / Tiny-ImageNet style)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # ResNet50 uses [3, 4, 6, 3] bottleneck blocks per stage
        self.layer1 = self.make_block(mid_channels=64,  num_blocks=3, stride=1)
        self.layer2 = self.make_block(mid_channels=128, num_blocks=4, stride=2)
        self.layer3 = self.make_block(mid_channels=256, num_blocks=6, stride=2)
        self.layer4 = self.make_block(mid_channels=512, num_blocks=3, stride=2)
        # Final stage outputs 512 * 4 = 2048 channels
        self.linear = nn.Linear(512 * BottleneckBlock.expansion, num_classes)

    def make_block(self, mid_channels, num_blocks, stride):
        layers = []
        # Only the first block of each stage downsamples
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(BottleneckBlock(self.in_channels, mid_channels, s))
            self.in_channels = mid_channels * BottleneckBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x