import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ShallowResNet(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64),
        )

        # self.layer2 = nn.Sequential(
        #     BasicBlock(64, 128, stride=2),
        #     BasicBlock(128, 128),
        # )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(128, num_classes)
        self.fc = nn.Linear(64, num_classes)

        self.layer1[0].bn1 = nn.LayerNorm([64, 8, 8], elementwise_affine=False)
        self.layer1[0].bn2 = nn.LayerNorm([64, 8, 8], elementwise_affine=False)
        self.layer1[1].bn1 = nn.LayerNorm([64, 8, 8], elementwise_affine=False)
        self.layer1[1].bn2 = nn.LayerNorm([64, 8, 8], elementwise_affine=False)
        # self.layer2[0].bn1 = nn.LayerNorm([128, 4, 4], elementwise_affine=False)
        # self.layer2[0].bn2 = nn.LayerNorm([128, 4, 4], elementwise_affine=False)
        # self.layer2[0].downsample[1] = nn.LayerNorm([128, 4, 4], elementwise_affine=False)
        # self.layer2[1].bn1 = nn.LayerNorm([128, 4, 4], elementwise_affine=False)
        # self.layer2[1].bn2 = nn.LayerNorm([128, 4, 4], elementwise_affine=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        # out = self.layer2(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
