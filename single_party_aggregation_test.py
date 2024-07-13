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
        print(f"Train Epoch: {epoch}/{epochs} \tAverage Loss: {avg_loss:.6f}")


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
    accuracy = correct / len(test_loader.dataset)
    return test_loss, accuracy


def prune_cnn(original_cnn, dropout_rate=0.5):
    scale_factor = 1 / (1 - dropout_rate)

    conv1 = original_cnn.layer1[0]
    num_output_channels_to_prune_conv1 = int(conv1.out_channels * dropout_rate)
    output_indices_to_prune_conv1 = np.random.choice(
        conv1.out_channels, num_output_channels_to_prune_conv1, replace=False
    )
    indices_to_prune_conv1 = {"output": output_indices_to_prune_conv1}
    pruned_layer1 = prune_conv_layer(conv1, indices_to_prune_conv1)
    pruned_layer1.weight.data *= scale_factor

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
    pruned_layer2.weight.data *= scale_factor

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
    pruned_layer3.weight.data *= scale_factor

    fc = original_cnn.fc
    input_indices_to_prune_fc = []
    for channel_index in output_indices_to_prune_conv3:
        start_index = channel_index * 4 * 4
        end_index = (channel_index + 1) * 4 * 4
        input_indices_to_prune_fc.extend(list(range(start_index, end_index)))
    input_indices_to_prune_fc = np.sort(input_indices_to_prune_fc)
    indices_to_prune_fc = {"input": input_indices_to_prune_fc}
    pruned_fc = prune_linear_layer(fc, indices_to_prune_fc)
    pruned_fc.weight.data *= scale_factor

    pruned_cnn = CNN()
    pruned_cnn.layer1[0] = pruned_layer1
    pruned_cnn.layer2[0] = pruned_layer2
    pruned_cnn.layer3[0] = pruned_layer3
    pruned_cnn.fc = pruned_fc

    return (
        pruned_cnn,
        indices_to_prune_conv1,
        indices_to_prune_conv2,
        indices_to_prune_conv3,
        indices_to_prune_fc,
    )


def aggregate_cnn(
    original_cnn, pruned_cnn, num_samples, dropout_rate, **indices_to_prune
):
    indices_to_prune_conv1 = indices_to_prune["indices_to_prune_conv1"]
    indices_to_prune_conv2 = indices_to_prune["indices_to_prune_conv2"]
    indices_to_prune_conv3 = indices_to_prune["indices_to_prune_conv3"]
    indices_to_prune_fc = indices_to_prune["indices_to_prune_fc"]

    scale_factor = 1 - dropout_rate
    pruned_cnn.layer1[0].weight.data *= scale_factor
    pruned_cnn.layer2[0].weight.data *= scale_factor
    pruned_cnn.layer3[0].weight.data *= scale_factor
    pruned_cnn.fc.weight.data *= scale_factor

    aggregate_conv_layers(
        original_cnn.layer1[0],
        [
            pruned_cnn.layer1[0],
        ],
        [
            indices_to_prune_conv1,
        ],
        [
            num_samples,
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
            num_samples,
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
            num_samples,
        ],
    )
    aggregate_linear_layers(
        original_cnn.fc,
        [
            pruned_cnn.fc,
        ],
        [
            indices_to_prune_fc,
        ],
        [
            num_samples,
        ],
    )


ROUNDS = 100
BATCH_SIZE = 128
EPOCHS = 5
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

# Test the original model
original_cnn.to(device)
test_loss_original_cnn, accuracy_original_cnn = test(
    original_cnn, device, test_loader, criterion
)
print(
    f"Original model test loss: {test_loss_original_cnn:.6f}, accuracy: {accuracy_original_cnn:.6f}"
)
print("-" * 80)

for round in range(ROUNDS):
    print(f"Round {round + 1}/{ROUNDS}")

    # Prune a model each round
    dropout_rate = 0.5

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
    test_loss_pruned_cnn, accuracy_pruned_cnn = test(
        pruned_cnn, device, test_loader, criterion
    )

    # # Make all the parameters of the original CNN zero
    # for param in original_cnn.parameters():
    #     param.data = torch.zeros_like(param.data, device=param.device)

    # Aggregation
    aggregate_cnn(
        original_cnn,
        pruned_cnn,
        len(train_dataset),
        dropout_rate,
        indices_to_prune_conv1=indices_to_prune_conv1,
        indices_to_prune_conv2=indices_to_prune_conv2,
        indices_to_prune_conv3=indices_to_prune_conv3,
        indices_to_prune_fc=indices_to_prune_fc,
    )

    # Test the aggregated model
    original_cnn.to(device)
    test_loss_aggregated, accuracy_aggregated = test(
        original_cnn, device, test_loader, criterion
    )

    print(
        f"Pruned model test loss: {test_loss_pruned_cnn:.6f}, accuracy: {accuracy_pruned_cnn:.6f}"
    )
    print(
        f"Aggregated model test loss: {test_loss_aggregated:.6f}, accuracy: {accuracy_aggregated:.6f}"
    )
    print("-" * 80)
