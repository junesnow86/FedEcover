import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from modules.models import CNN
from modules.utils import (
    aggregate_conv_layers,
    aggregate_linear_layers,
    calculate_model_size,
    # federated_averaging,
    prune_conv_layer,
    prune_linear_layer,
)

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
    root="./data", train=True, transform=transform
)
# train_loader = DataLoader(
#     train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
# )

test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, transform=transform
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

# Split the training set into two subsets
targets = train_dataset.targets
class_subset1 = set(range(5))
class_subset2 = set(range(5, 10))
indices_subset1 = [i for i, t in enumerate(targets) if t in class_subset1]
indices_subset2 = [i for i, t in enumerate(targets) if t in class_subset2]
subset1 = Subset(train_dataset, indices_subset1)
subset2 = Subset(train_dataset, indices_subset2)
train_loader1 = DataLoader(subset1, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
train_loader2 = DataLoader(subset2, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()

central_model = CNN()
participant_model1 = CNN()
participant_model2 = CNN()
participant_model1.load_state_dict(central_model.state_dict())
participant_model2.load_state_dict(central_model.state_dict())

# Pruning
participant_model1_layer1_0, participant_model1_layer1_0_pruned_indices = (
    prune_conv_layer(central_model.layer1[0], 0.8, prune_input=False)
)
participant_model1.layer1[1] = nn.BatchNorm2d(
    participant_model1_layer1_0.weight.shape[0]
)
participant_model1.layer1[0] = participant_model1_layer1_0

participant_model1_layer2_0, participant_model1_layer2_0_pruned_indices = (
    prune_conv_layer(central_model.layer2[0], 0.8)
)
participant_model1.layer2[1] = nn.BatchNorm2d(
    participant_model1_layer2_0.weight.shape[0]
)
participant_model1.layer2[0] = participant_model1_layer2_0

participant_model1_layer3_0, participant_model1_layer3_0_pruned_indices = (
    prune_conv_layer(central_model.layer3[0], 0.8)
)
participant_model1.layer3[1] = nn.BatchNorm2d(
    participant_model1_layer3_0.weight.shape[0]
)
participant_model1.layer3[0] = participant_model1_layer3_0

participant_model1.fc, participant_model1_fc_pruned_indices = prune_linear_layer(
    central_model.fc,
    prune_input=True,
    prune_output=False,
    num_neurons_to_prune={
        "input": (
            central_model.layer3[0].weight.shape[0]
            - participant_model1_layer3_0.weight.shape[0]
        )
        * 4
        * 4
    },
    use_absolute_number=True,
)


participant_model2_layer1_0, participant_model2_layer1_0_pruned_indices = (
    prune_conv_layer(central_model.layer1[0], 0.2, prune_input=False)
)
participant_model2.layer1[1] = nn.BatchNorm2d(
    participant_model2_layer1_0.weight.shape[0]
)
participant_model2.layer1[0] = participant_model2_layer1_0

participant_model2_layer2_0, participant_model2_layer2_0_pruned_indices = (
    prune_conv_layer(central_model.layer2[0], 0.2)
)
participant_model2.layer2[1] = nn.BatchNorm2d(
    participant_model2_layer2_0.weight.shape[0]
)
participant_model2.layer2[0] = participant_model2_layer2_0

participant_model2_layer3_0, participant_model2_layer3_0_pruned_indices = (
    prune_conv_layer(central_model.layer3[0], 0.2)
)
participant_model2.layer3[1] = nn.BatchNorm2d(
    participant_model2_layer3_0.weight.shape[0]
)
participant_model2.layer3[0] = participant_model2_layer3_0

participant_model2.fc, participant_model2_fc_pruned_indices = prune_linear_layer(
    central_model.fc,
    prune_output=False,
    num_neurons_to_prune={
        "input": (
            central_model.layer3[0].weight.shape[0]
            - participant_model2_layer3_0.weight.shape[0]
        )
        * 4
        * 4
    },
    use_absolute_number=True,
)

calculate_model_size(central_model)
calculate_model_size(participant_model1)
calculate_model_size(participant_model2)

central_model.to(device)
participant_model1.to(device)
participant_model2.to(device)

# Local Training
optimizer = optim.Adam(participant_model1.parameters(), lr=LEARNING_RATE)
for epoch in range(EPOCHS):
    participant_model1.train()
    running_loss = 0.0
    for images, labels in train_loader1:
        images = images.to(device)
        labels = labels.to(device)

        outputs = participant_model1(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss / len(train_loader1):.4f}"
    )

optimizer = optim.Adam(participant_model2.parameters(), lr=LEARNING_RATE)
for epoch in range(EPOCHS):
    participant_model2.train()
    running_loss = 0.0
    for images, labels in train_loader2:
        images = images.to(device)
        labels = labels.to(device)

        outputs = participant_model2(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss / len(train_loader2):.4f}"
    )

# Testing
participant_model1.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = participant_model1(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy of the participant model1 on the test images: {accuracy:.2f}%")

participant_model2.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = participant_model2(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy of the participant model2 on the test images: {accuracy:.2f}%")

# Aggregation
# avg_weights = federated_averaging(
#     [participant_model1.state_dict(), participant_model2.state_dict()],
#     [len(subset1), len(subset2)],
# )
# central_model.load_state_dict(avg_weights)

# Testing
central_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = central_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(
    f"Accuracy of the central model on the test images before aggregation: {accuracy:.2f}%"
)

aggregate_conv_layers(
    central_model.layer1[0],
    [participant_model1.layer1[0], participant_model2.layer1[0]],
    [
        participant_model1_layer1_0_pruned_indices,
        participant_model2_layer1_0_pruned_indices,
    ],
    [len(subset1), len(subset2)],
)
aggregate_conv_layers(
    central_model.layer2[0],
    [participant_model1.layer2[0], participant_model2.layer2[0]],
    [
        participant_model1_layer2_0_pruned_indices,
        participant_model2_layer2_0_pruned_indices,
    ],
    [len(subset1), len(subset2)],
)
aggregate_conv_layers(
    central_model.layer3[0],
    [participant_model1.layer3[0], participant_model2.layer3[0]],
    [
        participant_model1_layer3_0_pruned_indices,
        participant_model2_layer3_0_pruned_indices,
    ],
    [len(subset1), len(subset2)],
)
aggregate_linear_layers(
    central_model.fc,
    [participant_model1.fc, participant_model2.fc],
    [participant_model1_fc_pruned_indices, participant_model2_fc_pruned_indices],
    [len(subset1), len(subset2)],
)

# Testing
central_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = central_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy of the central model on the test images: {accuracy:.2f}%")
