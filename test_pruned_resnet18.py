import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

from modules.utils import test, train, calculate_model_size
from modules.pruning import prune_resnet18

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

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(weights=None)
calculate_model_size(model)

model, _ = prune_resnet18(model, 0.5)
calculate_model_size(model)

optimizer = optim.Adam(model.parameters(), lr=LR)
_ = train(model, optimizer, criterion, train_loader, device, EPOCHS, print_log=True)
test_loss, test_acc, _ = test(model, criterion, test_loader, device, num_classes=10)
print(f"Test Loss: {test_loss:.6f}, Test Accuracy: {test_acc}")
