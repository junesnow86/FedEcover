import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, dataset: str) -> None:
        super().__init__()

        if dataset == "cifar10":
            num_classes = 10
            in_channels = 3
            fc_size = 4
        elif dataset == "cifar100":
            num_classes = 100
            in_channels = 3
            fc_size = 4
        elif dataset == "celeba":
            num_classes = 2
            in_channels = 3
            fc_size = 16
        else:
            raise ValueError(f"Dataset {dataset} not supported")

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
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
        self.fc = nn.Linear(256 * fc_size * fc_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class FEMNISTCNN(nn.Module):
    def __init__(self, dataset):
        super().__init__()

        if dataset == "femnist":
            num_classes = 62
            in_channels = 1
            fc1_size = 7
        else:
            raise ValueError(f"Dataset {dataset} not supported")

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # self.fc1 = nn.Linear(64 * 7 * 7, 2048)
        self.fc1 = nn.Linear(64 * fc1_size * fc1_size, 2048)
        self.fc2 = nn.Linear(2048, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out
