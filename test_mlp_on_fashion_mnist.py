# Import necessary libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from modules.utils import aggregate_linear_layers, prune_linear_layer


# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the image
        return self.layers(x)


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


# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Fashion-MNIST dataset
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


# Instantiate the model and move it to the device (GPU if available)
model = MLP()

# Prune the model
dropout_rate = 0.8

linear1 = model.layers[0]
num_output_neurons_to_prune_linear1 = int(dropout_rate * linear1.out_features)
output_indices_to_prune_linear1 = np.sort(
    np.random.choice(
        linear1.out_features, num_output_neurons_to_prune_linear1, replace=False
    )
)
indices_to_prune_linear1 = {"output": output_indices_to_prune_linear1}
pruned_linear1 = prune_linear_layer(linear1, indices_to_prune_linear1)

linear2 = model.layers[2]
num_output_neurons_to_prune_linear2 = int(dropout_rate * linear2.out_features)
output_indices_to_prune_linear2 = np.sort(
    np.random.choice(
        linear2.out_features, num_output_neurons_to_prune_linear2, replace=False
    )
)
indices_to_prune_linear2 = {
    "input": output_indices_to_prune_linear1,
    "output": output_indices_to_prune_linear2,
}
pruned_linear2 = prune_linear_layer(linear2, indices_to_prune_linear2)

linear3 = model.layers[4]
indices_to_prune_linear3 = {"input": output_indices_to_prune_linear2}
pruned_linear3 = prune_linear_layer(linear3, indices_to_prune_linear3)

# Initialize the pruned model
pruned_model = MLP()
pruned_model.layers[0] = pruned_linear1
pruned_model.layers[2] = pruned_linear2
pruned_model.layers[4] = pruned_linear3

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pruned_model.parameters(), lr=0.001)


# Train and test the pruned model
pruned_model.to(device)
train(pruned_model, device, train_loader, optimizer, criterion)
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
aggregate_linear_layers(
    model.layers[0],
    [
        pruned_model.layers[0],
    ],
    [
        indices_to_prune_linear1,
    ],
    [
        len(train_dataset),
    ],
)
aggregate_linear_layers(
    model.layers[2],
    [
        pruned_model.layers[2],
    ],
    [
        indices_to_prune_linear2,
    ],
    [
        len(train_dataset),
    ],
)
aggregate_linear_layers(
    model.layers[4],
    [
        pruned_model.layers[4],
    ],
    [
        indices_to_prune_linear3,
    ],
    [
        len(train_dataset),
    ],
)

# Test the aggregated model
model.to(device)
print("Testing the aggregated model")
test(model, device, test_loader, criterion)
