import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(256, num_classes)
        self.fc = nn.Linear(256 * 4 * 4, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
