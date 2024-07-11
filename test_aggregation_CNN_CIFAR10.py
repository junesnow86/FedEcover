import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from modules.models import CNN
from modules.utils import (
    aggregate_conv_layers,
    aggregate_linear_layers,
    prune_conv_layer,
    prune_linear_layer,
)

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=False, transform=transform
)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=False, transform=transform
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

original_cnn = CNN()

# Evaluate the original model
original_cnn.to(device)
original_cnn.eval()
correct = 0
total = 0
# with torch.no_grad():
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = original_cnn(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# accuracy = 100 * correct / total
# print(f"Accuracy of the original model on the test images: {accuracy:.2f}%")

dropout_rate = 0.5

layer1_output_channels_to_prune = int(
    original_cnn.layer1[0].out_channels * dropout_rate
)
layer1_output_indices_to_prune = np.sort(
    np.random.choice(
        range(original_cnn.layer1[0].out_channels),
        layer1_output_channels_to_prune,
        replace=False,
    )
)
layer1_pruned_indices = {"output": layer1_output_indices_to_prune}
pruned_layer1 = prune_conv_layer(original_cnn.layer1[0], layer1_pruned_indices)

layer2_output_channels_to_prune = int(
    original_cnn.layer2[0].out_channels * dropout_rate
)
layer2_output_indices_to_prune = np.sort(
    np.random.choice(
        range(original_cnn.layer2[0].out_channels),
        layer2_output_channels_to_prune,
        replace=False,
    )
)
layer2_pruned_indices = {
    "input": layer1_output_indices_to_prune,
    "output": layer2_output_indices_to_prune,
}
pruned_layer2 = prune_conv_layer(original_cnn.layer2[0], layer2_pruned_indices)

layer3_output_channels_to_prune = int(
    original_cnn.layer3[0].out_channels * dropout_rate
)
layer3_output_indices_to_prune = np.sort(
    np.random.choice(
        range(original_cnn.layer3[0].out_channels),
        layer3_output_channels_to_prune,
        replace=False,
    )
)
layer3_pruned_indices = {
    "input": layer2_output_indices_to_prune,
    "output": layer3_output_indices_to_prune,
}
pruned_layer3 = prune_conv_layer(original_cnn.layer3[0], layer3_pruned_indices)

fc_pruned_indices = []
for channel_index in layer3_output_indices_to_prune:
    start_index = channel_index * 4 * 4
    end_index = (channel_index + 1) * 4 * 4
    fc_pruned_indices.extend(list(range(start_index, end_index)))
fc_input_indices_to_prune = np.sort(fc_pruned_indices)
pruned_fc = prune_linear_layer(original_cnn.fc, {"input": fc_input_indices_to_prune})

pruned_cnn = CNN()
pruned_cnn.layer1[0] = nn.Conv2d(
    in_channels=pruned_layer1.in_channels,
    out_channels=pruned_layer1.out_channels,
    kernel_size=pruned_layer1.kernel_size,
    stride=pruned_layer1.stride,
    padding=pruned_layer1.padding,
    dilation=pruned_layer1.dilation,
    groups=pruned_layer1.groups,
    bias=(pruned_layer1.bias is not None),
)
pruned_cnn.layer1[1] = nn.BatchNorm2d(layer1_output_channels_to_prune)
pruned_cnn.layer2[0] = nn.Conv2d(
    in_channels=pruned_layer2.in_channels,
    out_channels=pruned_layer2.out_channels,
    kernel_size=pruned_layer2.kernel_size,
    stride=pruned_layer2.stride,
    padding=pruned_layer2.padding,
    dilation=pruned_layer2.dilation,
    groups=pruned_layer2.groups,
    bias=(pruned_layer2.bias is not None),
)
pruned_cnn.layer2[1] = nn.BatchNorm2d(layer2_output_channels_to_prune)
pruned_cnn.layer3[0] = nn.Conv2d(
    in_channels=pruned_layer3.in_channels,
    out_channels=pruned_layer3.out_channels,
    kernel_size=pruned_layer3.kernel_size,
    stride=pruned_layer3.stride,
    padding=pruned_layer3.padding,
    dilation=pruned_layer3.dilation,
    groups=pruned_layer3.groups,
    bias=(pruned_layer3.bias is not None),
)
pruned_cnn.layer3[1] = nn.BatchNorm2d(layer3_output_channels_to_prune)
pruned_cnn.fc = nn.Linear(pruned_fc.in_features, pruned_fc.out_features)


pruned_cnn.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pruned_cnn.parameters(), lr=LEARNING_RATE)

# Training
for epoch in range(EPOCHS):
    pruned_cnn.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = pruned_cnn(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss / len(train_loader):.4f}")


# Aggregation
original_cnn.layer1[0].weight.data = torch.zeros_like(
    original_cnn.layer1[0].weight.data, device=original_cnn.layer1[0].weight.device
)
original_cnn.layer1[0].bias.data = torch.zeros_like(
    original_cnn.layer1[0].bias.data, device=original_cnn.layer1[0].bias.device
)
original_cnn.layer2[0].weight.data = torch.zeros_like(
    original_cnn.layer2[0].weight.data, device=original_cnn.layer2[0].weight.device
)
original_cnn.layer2[0].bias.data = torch.zeros_like(
    original_cnn.layer2[0].bias.data, device=original_cnn.layer2[0].bias.device
)
original_cnn.layer3[0].weight.data = torch.zeros_like(
    original_cnn.layer3[0].weight.data, device=original_cnn.layer3[0].weight.device
)
original_cnn.layer3[0].bias.data = torch.zeros_like(
    original_cnn.layer3[0].bias.data, device=original_cnn.layer3[0].bias.device
)
original_cnn.fc.weight.data = torch.zeros_like(
    original_cnn.fc.weight.data, device=original_cnn.fc.weight.device
)
original_cnn.fc.bias.data = torch.zeros_like(
    original_cnn.fc.bias.data, device=original_cnn.fc.bias.device
)

aggregate_conv_layers(
    original_cnn.layer1[0], [pruned_cnn.layer1[0]], [layer1_pruned_indices], [1]
)
aggregate_conv_layers(
    original_cnn.layer2[0], [pruned_cnn.layer2[0]], [layer2_pruned_indices], [1]
)
aggregate_conv_layers(
    original_cnn.layer3[0], [pruned_cnn.layer3[0]], [layer3_pruned_indices], [1]
)
aggregate_linear_layers(
    original_cnn.fc, [pruned_cnn.fc], [{"input": fc_input_indices_to_prune}], [1]
)

original_cnn.to(device)
pruned_cnn.eval()
original_cnn.eval()

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = pruned_cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy of the pruned model on the test images: {accuracy:.2f}%")

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = original_cnn(images)
        pruned_cnn_outputs = pruned_cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy of the aggregated model on the test images: {accuracy:.2f}%")

# Check inequality
with torch.no_grad():
    for images, labels in test_loader:
        images = images[0].unsqueeze(0).to(device)
        labels = labels[0].unsqueeze(0).to(device)
        outputs = original_cnn(images)
        pruned_cnn_outputs = pruned_cnn(images)
        assert torch.equal(
            outputs, pruned_cnn_outputs
        ), f"Original CNN output: \n{outputs}\nPruned CNN output: \n{pruned_cnn_outputs}"
        break
