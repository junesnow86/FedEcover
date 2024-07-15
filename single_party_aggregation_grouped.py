import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from modules.models import CNN
from modules.utils import (
    aggregate_cnn,
    prune_cnn_into_groups,
    test,
    train,
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

original_cnn = CNN()

# Test the original model
original_cnn.to(device)
test_loss_original_cnn, accuracy_original_cnn, _ = test(
    original_cnn, device, test_loader, criterion
)
print(
    f"Original model test loss: {test_loss_original_cnn:.6f}, accuracy: {accuracy_original_cnn:.6f}"
)
print("-" * 80)

dropout_rate = 0.8
num_groups = max(1, int(1 / (1 - dropout_rate)))
pruned_models = []
indices_to_prune_list = []

print(f"Number of groups: {num_groups}")

for round in range(ROUNDS):
    print(f"Round {round + 1}/{ROUNDS}")

    if round % num_groups == 0:
        pruned_models, indices_to_prune_list = prune_cnn_into_groups(
            original_cnn, dropout_rate
        )

    pruned_cnn = pruned_models[round % num_groups]
    indices_to_prune_conv1 = indices_to_prune_list[round % num_groups][
        "indices_to_prune_conv1"
    ]
    indices_to_prune_conv2 = indices_to_prune_list[round % num_groups][
        "indices_to_prune_conv2"
    ]
    indices_to_prune_conv3 = indices_to_prune_list[round % num_groups][
        "indices_to_prune_conv3"
    ]
    indices_to_prune_fc = indices_to_prune_list[round % num_groups][
        "indices_to_prune_fc"
    ]

    optimizer = optim.Adam(pruned_cnn.parameters(), lr=LEARNING_RATE)

    # Train and test the pruned model
    pruned_cnn.to(device)
    train(pruned_cnn, device, train_loader, optimizer, criterion, epochs=EPOCHS)
    test_loss_pruned_cnn, accuracy_pruned_cnn, _ = test(
        pruned_cnn, device, test_loader, criterion
    )
    print(f"Pruned model test accuracy: {accuracy_pruned_cnn:.6f}")

    # Aggregation
    if round % num_groups == num_groups - 1:
        print("*" * 80)
        print("Aggregating models")

        indices_to_prune_dict = {
            "indices_to_prune_conv1": [],
            "indices_to_prune_conv2": [],
            "indices_to_prune_conv3": [],
            "indices_to_prune_fc": [],
        }
        for i in range(num_groups):
            indices_to_prune_dict["indices_to_prune_conv1"].append(
                indices_to_prune_list[i]["indices_to_prune_conv1"]
            )
            indices_to_prune_dict["indices_to_prune_conv2"].append(
                indices_to_prune_list[i]["indices_to_prune_conv2"]
            )
            indices_to_prune_dict["indices_to_prune_conv3"].append(
                indices_to_prune_list[i]["indices_to_prune_conv3"]
            )
            indices_to_prune_dict["indices_to_prune_fc"].append(
                indices_to_prune_list[i]["indices_to_prune_fc"]
            )

        aggregate_cnn(
            original_cnn,
            pruned_models,
            [len(train_dataset)] * num_groups,
            [dropout_rate] * num_groups,
            indices_to_prune_dict,
        )

        # Test the aggregated model
        original_cnn.to(device)
        test_loss_aggregated, accuracy_aggregated, _ = test(
            original_cnn, device, test_loader, criterion
        )

        print(f"Aggregated model test accuracy: {accuracy_aggregated:.6f}")
        print("*" * 80)
    print("-" * 80)
