# import copy

# import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from modules.models import CNN
from modules.utils import (
    aggregate_cnn,
    # aggregate_conv_layers,
    # aggregate_linear_layers,
    # prune_conv_layer,
    # prune_linear_layer,
    prune_cnn,
)


# Training function
def train(model, device, train_loader, optimizer, criterion, epochs=30):
    original_device = next(model.parameters()).device
    model.to(device)
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

        correct = 0
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        avg_loss = total_loss / len(train_loader)
        print(
            f"Train Epoch: {epoch}\tAverage Loss: {avg_loss:.6f}\tAccuracy: {correct}/{len(train_loader.dataset)} ({100. * correct / len(train_loader.dataset):.0f}%)"
        )

    model.to(original_device)


# Testing function
def test(model, device, test_loader, criterion):
    original_device = next(model.parameters()).device
    model.to(device)
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

    test_loss /= len(test_loader)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n"
    )

    model.to(original_device)


BATCH_SIZE = 128
EPOCHS = 20
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
dropout_rate = 0.8

# conv1 = copy.deepcopy(original_cnn.layer1[0])
# num_output_channels_to_prune_conv1 = int(conv1.out_channels * dropout_rate)
# output_indices_to_prune_conv1 = np.random.choice(
#     conv1.out_channels, num_output_channels_to_prune_conv1, replace=False
# )
# indices_to_prune_conv1 = {"output": output_indices_to_prune_conv1}
# pruned_layer1 = prune_conv_layer(conv1, indices_to_prune_conv1)

# conv2 = copy.deepcopy(original_cnn.layer2[0])
# num_output_channels_to_prune_conv2 = int(conv2.out_channels * dropout_rate)
# output_indices_to_prune_conv2 = np.random.choice(
#     conv2.out_channels, num_output_channels_to_prune_conv2, replace=False
# )
# indices_to_prune_conv2 = {
#     "input": output_indices_to_prune_conv1,
#     "output": output_indices_to_prune_conv2,
# }
# pruned_layer2 = prune_conv_layer(conv2, indices_to_prune_conv2)

# conv3 = copy.deepcopy(original_cnn.layer3[0])
# num_output_channels_to_prune_conv3 = int(conv3.out_channels * dropout_rate)
# output_indices_to_prune_conv3 = np.random.choice(
#     conv3.out_channels, num_output_channels_to_prune_conv3, replace=False
# )
# indices_to_prune_conv3 = {
#     "input": output_indices_to_prune_conv2,
#     "output": output_indices_to_prune_conv3,
# }
# pruned_layer3 = prune_conv_layer(conv3, indices_to_prune_conv3)

# fc = copy.deepcopy(original_cnn.fc)
# input_indices_to_prune_fc = []
# for channel_index in output_indices_to_prune_conv3:
#     start_index = channel_index * 4 * 4
#     end_index = (channel_index + 1) * 4 * 4
#     input_indices_to_prune_fc.extend(list(range(start_index, end_index)))
# input_indices_to_prune_fc = np.sort(input_indices_to_prune_fc)
# indices_to_prune_fc = {"input": input_indices_to_prune_fc}
# pruned_fc = prune_linear_layer(fc, indices_to_prune_fc)

# pruned_cnn = CNN()
# pruned_cnn.layer1[0] = pruned_layer1
# pruned_cnn.layer2[0] = pruned_layer2
# pruned_cnn.layer3[0] = pruned_layer3
# pruned_cnn.fc = pruned_fc

(
    pruned_cnn,
    indices_to_prune_conv1,
    indices_to_prune_conv2,
    indices_to_prune_conv3,
    indices_to_prune_fc,
) = prune_cnn(original_cnn, dropout_rate=dropout_rate)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pruned_cnn.parameters(), lr=LEARNING_RATE)

# Train and test the pruned model
train(pruned_cnn, device, train_loader, optimizer, criterion, epochs=EPOCHS)
print("Testing the pruned model")
test(pruned_cnn, device, test_loader, criterion)

# Test the original model
print("Testing the original model")
test(original_cnn, device, test_loader, criterion)

# # Make all the parameters of the original CNN zero
# for param in original_cnn.parameters():
#     param.data = torch.zeros_like(param.data, device=param.device)

# Aggregation
# aggregate_conv_layers(
#     original_cnn.layer1[0],
#     [
#         pruned_cnn.layer1[0],
#     ],
#     [
#         indices_to_prune_conv1,
#     ],
#     [
#         len(train_dataset),
#     ],
# )
# aggregate_conv_layers(
#     original_cnn.layer2[0],
#     [
#         pruned_cnn.layer2[0],
#     ],
#     [
#         indices_to_prune_conv2,
#     ],
#     [
#         len(train_dataset),
#     ],
# )
# aggregate_conv_layers(
#     original_cnn.layer3[0],
#     [
#         pruned_cnn.layer3[0],
#     ],
#     [
#         indices_to_prune_conv3,
#     ],
#     [
#         len(train_dataset),
#     ],
# )
# aggregate_linear_layers(
#     original_cnn.fc,
#     [
#         pruned_cnn.fc,
#     ],
#     [
#         indices_to_prune_fc,
#     ],
#     [
#         len(train_dataset),
#     ],
# )
aggregate_cnn(
    original_cnn,
    [pruned_cnn],
    [len(train_dataset)],
    [dropout_rate],
    {
        "indices_to_prune_conv1": [indices_to_prune_conv1],
        "indices_to_prune_conv2": [indices_to_prune_conv2],
        "indices_to_prune_conv3": [indices_to_prune_conv3],
        "indices_to_prune_fc": [indices_to_prune_fc],
    },
)

# Test the aggregated model
print("Testing the aggregated model")
test(original_cnn, device, test_loader, criterion)

original_cnn.eval()
dropout = nn.Dropout()
