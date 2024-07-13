"""
Implement the test for zero-footprint dropout aggregation with two heterogeneous parties.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from modules.models import CNN
from modules.utils import aggregate_cnn, prune_cnn, test, train

ROUNDS = 20
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001

# ---------------------- Prepare the dataset ----------------------
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

# train_loader = DataLoader(
#     train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
# )
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

# Split the dataset into two parts
indices_subset1 = [i for i, (_, label) in enumerate(train_dataset) if label < 5]
indices_subset2 = [i for i, (_, label) in enumerate(train_dataset) if label >= 5]

subset1 = Subset(train_dataset, indices_subset1)
subset2 = Subset(train_dataset, indices_subset2)

train_loader1 = DataLoader(subset1, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
train_loader2 = DataLoader(subset2, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


# ---------------------- Prepare the global model ---------------------
global_cnn = CNN()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

# Test the global model
global_cnn.to(device)
test_loss_original_cnn, accuracy_original_cnn, class_accuracy_original_cnn = test(
    global_cnn, device, test_loader, criterion
)
print(
    f"Original model test loss: {test_loss_original_cnn:.6f}, accuracy: {accuracy_original_cnn:.6f}, class accuracy: {class_accuracy_original_cnn}"
)
print("-" * 80)


# ---------------------- Pruning, local training and aggregation ---------------------
for round in range(ROUNDS):
    print(f"Round {round + 1}/{ROUNDS}")

    # Prune two models with different dropout rates each round
    # ---------------------- Party 1 ---------------------
    dropout_rate1 = 0.5

    (
        pruned_cnn1,
        indices_to_prune_conv1_cnn1,
        indices_to_prune_conv2_cnn1,
        indices_to_prune_conv3_cnn1,
        indices_to_prune_fc_cnn1,
    ) = prune_cnn(global_cnn, dropout_rate=dropout_rate1)

    optimizer1 = torch.optim.Adam(pruned_cnn1.parameters(), lr=LEARNING_RATE)

    # Train and test the pruned model
    pruned_cnn1.to(device)
    train(pruned_cnn1, device, train_loader1, optimizer1, criterion, epochs=EPOCHS)
    test_loss_pruned_cnn1, accuracy_pruned_cnn1, class_accuracy_pruned_cnn1 = test(
        pruned_cnn1, device, test_loader, criterion
    )

    # ---------------------- Party 2 ---------------------
    dropout_rate2 = 0.5

    (
        pruned_cnn2,
        indices_to_prune_conv1_cnn2,
        indices_to_prune_conv2_cnn2,
        indices_to_prune_conv3_cnn2,
        indices_to_prune_fc_cnn2,
    ) = prune_cnn(global_cnn, dropout_rate=dropout_rate2)

    optimizer2 = torch.optim.Adam(pruned_cnn2.parameters(), lr=LEARNING_RATE)

    # Train and test the pruned model
    pruned_cnn2.to(device)
    train(pruned_cnn2, device, train_loader2, optimizer2, criterion, epochs=EPOCHS)
    test_loss_pruned_cnn2, accuracy_pruned_cnn2, class_accuracy_pruned_cnn2 = test(
        pruned_cnn2, device, test_loader, criterion
    )

    # Aggregation
    aggregate_cnn(
        global_cnn,
        [pruned_cnn1, pruned_cnn2],
        [len(subset1), len(subset2)],
        [dropout_rate1, dropout_rate2],
        {
            "indices_to_prune_conv1": [
                indices_to_prune_conv1_cnn1,
                indices_to_prune_conv1_cnn2,
            ],
            "indices_to_prune_conv2": [
                indices_to_prune_conv2_cnn1,
                indices_to_prune_conv2_cnn2,
            ],
            "indices_to_prune_conv3": [
                indices_to_prune_conv3_cnn1,
                indices_to_prune_conv3_cnn2,
            ],
            "indices_to_prune_fc": [indices_to_prune_fc_cnn1, indices_to_prune_fc_cnn2],
        },
    )

    # Test the aggregated model
    global_cnn.to(device)
    test_loss_aggregated, accuracy_aggregated, class_accuracy_aggregated = test(
        global_cnn, device, test_loader, criterion
    )

    print(
        f"Pruned model 1 test loss: {test_loss_pruned_cnn1:.6f}, accuracy: {accuracy_pruned_cnn1:.6f}, class accuracy: {class_accuracy_pruned_cnn1}"
    )
    print(
        f"Pruned model 2 test loss: {test_loss_pruned_cnn2:.6f}, accuracy: {accuracy_pruned_cnn2:.6f}, class accuracy: {class_accuracy_pruned_cnn2}"
    )
    print(
        f"Aggregated model test loss: {test_loss_aggregated:.6f}, accuracy: {accuracy_aggregated:.6f}, class accuracy: {class_accuracy_aggregated}"
    )
    print("-" * 80)
