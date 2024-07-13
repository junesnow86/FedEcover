import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from modules.models import CNN
from modules.utils import (
    prune_cnn,
    test,
    train,
)

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001

train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=False, transform=train_transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=False, transform=test_transform
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

original_cnn = CNN()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

for dropour_rate in np.arange(0.2, 1.2, 0.2):
    dropout_rate = 1 - dropour_rate

    (
        pruned_cnn,
        indices_to_prune_conv1,
        indices_to_prune_conv2,
        indices_to_prune_conv3,
        indices_to_prune_fc,
    ) = prune_cnn(original_cnn, dropout_rate=dropout_rate)

    optimizer = optim.Adam(pruned_cnn.parameters(), lr=LEARNING_RATE)

    # Train and test the pruned model
    pruned_cnn.to(device)
    train(pruned_cnn, device, train_loader, optimizer, criterion, epochs=EPOCHS)
    test_loss_pruned_cnn, accuracy_pruned_cnn, _ = test(
        pruned_cnn, device, test_loader, criterion
    )

    print(
        f"Pruned model with dropout_rate {dropout_rate} test accuracy: {accuracy_pruned_cnn:.6f}"
    )
