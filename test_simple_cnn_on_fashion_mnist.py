import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from modules.utils import (
    aggregate_conv_layers,
    aggregate_linear_layers,
    prune_conv_layer,
    prune_linear_layer,
)


class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


# Training function
def train(model, device, train_loader, optimizer, criterion, epochs=10):
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

train_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=False, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root="./data", train=False, download=False, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


model = FashionCNN()

# Prune the model
dropout_rate = 0.9

conv1 = model.layer1[0]
num_output_channels_to_prune_conv1 = int(dropout_rate * conv1.out_channels)
output_indices_to_prune_conv1 = np.sort(
    np.random.choice(
        conv1.out_channels, num_output_channels_to_prune_conv1, replace=False
    )
)
indices_to_prune_conv1 = {"output": output_indices_to_prune_conv1}
pruned_conv1 = prune_conv_layer(conv1, indices_to_prune_conv1)

conv2 = model.layer2[0]
num_output_channels_to_prune_conv2 = int(dropout_rate * conv2.out_channels)
output_indices_to_prune_conv2 = np.sort(
    np.random.choice(
        conv2.out_channels, num_output_channels_to_prune_conv2, replace=False
    )
)
indices_to_prune_conv2 = {
    "input": output_indices_to_prune_conv1,
    "output": output_indices_to_prune_conv2,
}
pruned_conv2 = prune_conv_layer(conv2, indices_to_prune_conv2)

fc1 = model.fc1
input_indices_to_prune_fc1 = []
for channel_index in output_indices_to_prune_conv2:
    start_index = channel_index * 7 * 7
    end_index = (channel_index + 1) * 7 * 7
    input_indices_to_prune_fc1.extend(list(range(start_index, end_index)))
input_indices_to_prune_fc1 = np.sort(input_indices_to_prune_fc1)
num_output_neurons_to_prune_fc1 = int(dropout_rate * fc1.out_features)
output_indices_to_prune_fc1 = np.sort(
    np.random.choice(fc1.out_features, num_output_neurons_to_prune_fc1, replace=False)
)
indices_to_prune_fc1 = {
    "input": input_indices_to_prune_fc1,
    "output": output_indices_to_prune_fc1,
}
pruned_fc1 = prune_linear_layer(fc1, indices_to_prune_fc1)

fc2 = model.fc2
indices_to_prune_fc2 = {"input": output_indices_to_prune_fc1}
pruned_fc2 = prune_linear_layer(fc2, indices_to_prune_fc2)

# Initialize the pruned model
pruned_model = FashionCNN()
pruned_model.layer1[0] = pruned_conv1
pruned_model.layer2[0] = pruned_conv2
pruned_model.fc1 = pruned_fc1
pruned_model.fc2 = pruned_fc2

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pruned_model.parameters(), lr=0.001)

# Train and test the pruned model
pruned_model.to(device)
train(pruned_model, device, train_loader, optimizer, criterion, epochs=5)
print("Testing the pruned model")
test(pruned_model, device, test_loader, criterion)

# Test the original model
model.to(device)
print("Testing the original model")
test(model, device, test_loader, criterion)

# Make all the parameters of the original model zeros
for param in model.parameters():
    param.data = torch.zeros_like(param.data)

# Aggregate the pruned model parameters to the original model
aggregate_conv_layers(
    model.layer1[0],
    [
        pruned_model.layer1[0],
    ],
    [
        indices_to_prune_conv1,
    ],
    [
        len(train_dataset),
    ],
)
aggregate_conv_layers(
    model.layer2[0],
    [
        pruned_model.layer2[0],
    ],
    [
        indices_to_prune_conv2,
    ],
    [
        len(train_dataset),
    ],
)
aggregate_linear_layers(
    model.fc1,
    [
        pruned_model.fc1,
    ],
    [
        indices_to_prune_fc1,
    ],
    [
        len(train_dataset),
    ],
)
aggregate_linear_layers(
    model.fc2,
    [
        pruned_model.fc2,
    ],
    [
        indices_to_prune_fc2,
    ],
    [
        len(train_dataset),
    ],
)

# Test the aggregated model
model.to(device)
print("Testing the aggregated model")
test(model, device, test_loader, criterion)
