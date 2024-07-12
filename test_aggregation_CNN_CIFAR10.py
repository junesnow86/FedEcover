from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from modules.utils import (
    aggregate_conv_layers,
    aggregate_linear_layers,
    prune_conv_layer,
    prune_linear_layer,
)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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
        self.fc = nn.Linear(256 * 4 * 4, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# Training function
def train(model, device, train_loader, optimizer, criterion, epochs=30):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Train Epoch: {epoch} \tAverage Loss: {avg_loss:.6f}")


# Testing function
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n"
    )


BATCH_SIZE = 128
EPOCHS = 10
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

original_cnn = CNN()

# Prune the model
dropout_rate = 0.5

conv1 = original_cnn.layer1[0]
num_output_channels_to_prune_conv1 = int(conv1.out_channels * dropout_rate)
output_indices_to_prune_conv1 = np.random.choice(
    conv1.out_channels, num_output_channels_to_prune_conv1, replace=False
)
indices_to_prune_conv1 = {"output": output_indices_to_prune_conv1}
pruned_layer1 = prune_conv_layer(conv1, indices_to_prune_conv1)

conv2 = original_cnn.layer2[0]
num_output_channels_to_prune_conv2 = int(conv2.out_channels * dropout_rate)
output_indices_to_prune_conv2 = np.random.choice(
    conv2.out_channels, num_output_channels_to_prune_conv2, replace=False
)
indices_to_prune_conv2 = {
    "input": output_indices_to_prune_conv1,
    "output": output_indices_to_prune_conv2,
}
pruned_layer2 = prune_conv_layer(conv2, indices_to_prune_conv2)

conv3 = original_cnn.layer3[0]
num_output_channels_to_prune_conv3 = int(conv3.out_channels * dropout_rate)
output_indices_to_prune_conv3 = np.random.choice(
    conv3.out_channels, num_output_channels_to_prune_conv3, replace=False
)
indices_to_prune_conv3 = {
    "input": output_indices_to_prune_conv2,
    "output": output_indices_to_prune_conv3,
}
pruned_layer3 = prune_conv_layer(conv3, indices_to_prune_conv3)

fc = original_cnn.fc
input_indices_to_prune_fc = []
for channel_index in output_indices_to_prune_conv3:
    start_index = channel_index * 4 * 4
    end_index = (channel_index + 1) * 4 * 4
    input_indices_to_prune_fc.extend(list(range(start_index, end_index)))
input_indices_to_prune_fc = np.sort(input_indices_to_prune_fc)
indices_to_prune_fc = {"input": input_indices_to_prune_fc}
pruned_fc = prune_linear_layer(fc, indices_to_prune_fc)

pruned_cnn = CNN()
pruned_cnn.layer1[0] = pruned_layer1
pruned_cnn.layer2[0] = pruned_layer2
pruned_cnn.layer3[0] = pruned_layer3
pruned_cnn.fc = pruned_fc

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pruned_cnn.parameters(), lr=LEARNING_RATE)

# Train and test the pruned model
pruned_cnn.to(device)
train(pruned_cnn, device, train_loader, optimizer, criterion, epochs=EPOCHS)
print("Testing the pruned model")
test(pruned_cnn, device, test_loader, criterion)

# Test the original model
original_cnn.to(device)
print("Testing the original model")
test(original_cnn, device, test_loader, criterion)

# Make all the parameters of the original CNN zero
for param in original_cnn.parameters():
    param.data = torch.zeros_like(param.data, device=param.device)

# Aggregation
aggregate_conv_layers(
    original_cnn.layer1[0],
    [
        pruned_cnn.layer1[0],
    ],
    [
        indices_to_prune_conv1,
    ],
    [
        len(train_dataset),
    ],
)
aggregate_conv_layers(
    original_cnn.layer2[0],
    [
        pruned_cnn.layer2[0],
    ],
    [
        indices_to_prune_conv2,
    ],
    [
        len(train_dataset),
    ],
)
aggregate_conv_layers(
    original_cnn.layer3[0],
    [
        pruned_cnn.layer3[0],
    ],
    [
        indices_to_prune_conv3,
    ],
    [
        len(train_dataset),
    ],
)
start = time()
aggregate_linear_layers(
    original_cnn.fc,
    [
        pruned_cnn.fc,
    ],
    [
        indices_to_prune_fc,
    ],
    [
        len(train_dataset),
    ],
)
print(f"Time to aggregate the models: {time() - start:.4f}")

# Test the aggregated model
original_cnn.to(device)
print("Testing the aggregated model")
test(original_cnn, device, test_loader, criterion)
