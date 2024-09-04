import csv
import os
import pickle
import random
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

from modules.args_parser import get_args
from modules.constants import NORMALIZATION_STATS
from modules.models import CNN
from modules.pruning import prune_resnet18
from modules.utils import evaluate_acc, train

args = get_args()
SAVE_DIR = args.save_dir
MODEL_TYPE = args.model
DATASET = args.dataset
if DATASET == "cifar10":
    NUM_CLASSES = 10
elif DATASET == "cifar100":
    NUM_CLASSES = 100
else:
    raise ValueError(f"Dataset {DATASET} not supported.")
ROUNDS = args.round
EPOCHS = args.epochs
LR = args.lr
BATCH_SIZE = args.batch_size
NUM_CLIENTS = args.num_clients

seed = 18
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            NORMALIZATION_STATS[DATASET]["mean"], NORMALIZATION_STATS[DATASET]["std"]
        ),
    ]
)
if DATASET == "cifar10":
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=False, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=False, transform=transform
    )
elif DATASET == "cifar100":
    train_dataset = datasets.CIFAR100(
        root="./data", train=True, download=False, transform=transform
    )
    test_dataset = datasets.CIFAR100(
        root="./data", train=False, download=False, transform=transform
    )

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

if MODEL_TYPE == "cnn":
    model = CNN(num_classes=NUM_CLASSES)
elif MODEL_TYPE == "resnet":
    model = resnet18(num_classes=NUM_CLASSES, weights=None)
else:
    raise ValueError(f"Model {MODEL_TYPE} not supported.")
print(f"[Model Architecture]\n{model}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loss = train(model, optimizer, criterion, train_loader, device, EPOCHS)
val_result = evaluate_acc(model, test_loader, device)
print(
    f"Train Loss: {train_loss:.4f}\tValidation Loss: {val_result['loss']:.4f}\tValidation Accuracy: {val_result['accuracy']:.4f}"
)

# 写入 CSV 文件
# csv_file_path = "accuracy_records.csv"
# with open(csv_file_path, mode="w", newline="") as file:
#     writer = csv.writer(file)
#     writer.writerow(["Epoch", "Train Accuracy", "Validation Accuracy"])
#     for epoch, (train_acc, val_acc) in enumerate(
#         zip(train_acc_list, val_acc_list), start=1
#     ):
#         writer.writerow([epoch, train_acc, val_acc])

# print(f"Accuracy records have been written to {csv_file_path}")
