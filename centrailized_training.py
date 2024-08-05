import csv
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from modules.models import CNN
from modules.utils import test, train_and_validate

EPOCHS = 100
LR = 0.001
BATCH_SIZE = 128

seed = 18
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=False, transform=transform
)
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=False, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_acc_list, val_acc_list = train_and_validate(
    model, train_loader, test_loader, optimizer, criterion, EPOCHS, device
)
_, test_acc, class_acc = test(model, device, test_loader, criterion)
print(f"Test accuracy: {test_acc}\tClass-wise accuracy: {class_acc}")


# 写入 CSV 文件
csv_file_path = "accuracy_records.csv"
with open(csv_file_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train Accuracy", "Validation Accuracy"])
    for epoch, (train_acc, val_acc) in enumerate(
        zip(train_acc_list, val_acc_list), start=1
    ):
        writer.writerow([epoch, train_acc, val_acc])

print(f"Accuracy records have been written to {csv_file_path}")
