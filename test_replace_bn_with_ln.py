import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

from modules.utils import replace_bn_with_ln, replace_bn_with_sbn, test, train

EPOCHS = 20
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
replace_bn_with_sbn(model)
optimizer = optim.Adam(model.parameters(), lr=LR)
_ = train(model, optimizer, criterion, train_loader, device, EPOCHS)
test_loss, test_acc, _ = test(model, criterion, test_loader, device, num_classes=10)
print(f"Original Test Loss: {test_loss}\tTest accuracy: {test_acc}")
print("Trained bn1 parameters:")
try:
    print(model.bn1.weight)
    print(model.bn1.bias)
except Exception:
    print("No BN parameters in the model")
print("Running mean and variance of bn1:")
try:
    print(model.bn1.running_mean)
    print(model.bn1.running_var)
except Exception:
    print("No running mean and variance in the model")

model = resnet18(weights=None)
print("Replacing BN with LN")
replace_bn_with_ln(model, affine=False)
print(model.layer2[0].downsample[1])
optimizer = optim.Adam(model.parameters(), lr=LR)
_ = train(model, optimizer, criterion, train_loader, device, EPOCHS)
test_loss, test_acc, _ = test(model, criterion, test_loader, device, num_classes=10)
print(f"After replacing BN with LN, Test Loss: {test_loss}\tTest accuracy: {test_acc}")
print("Trained ln1 parameters:")
try:
    print(model.bn1.weight)
    print(model.bn1.bias)
except Exception:
    print("No LN parameters in the model")
