import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from modules.models import CNN
from modules.utils import (
    aggregate_cnn,
    prune_cnn,
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
test_loss_original_cnn, accuracy_original_cnn, _ = test(
    original_cnn, device, test_loader, criterion
)
print(
    f"Original model test loss: {test_loss_original_cnn:.6f}, accuracy: {accuracy_original_cnn:.6f}"
)
print("-" * 80)

dropout_rate = 0.2
accuracy_last_aggregated = 0.0

for round in range(ROUNDS):
    print(f"Round {round + 1}/{ROUNDS}")

    (
        pruned_cnn,
        indices_to_prune_conv1,
        indices_to_prune_conv2,
        indices_to_prune_conv3,
        indices_to_prune_fc,
    ) = prune_cnn(original_cnn, dropout_rate=dropout_rate)

    optimizer = optim.Adam(pruned_cnn.parameters(), lr=LEARNING_RATE)

    # Train and test the pruned model
    train(pruned_cnn, device, train_loader, optimizer, criterion, epochs=EPOCHS)
    test_loss_pruned_cnn, accuracy_pruned_cnn, _ = test(
        pruned_cnn, device, test_loader, criterion
    )
    print(f"Pruned model test accuracy: {accuracy_pruned_cnn:.6f}")

    # Aggregation
    original_cnn_backup = original_cnn.state_dict()
    aggregate_cnn(
        original_cnn,
        [pruned_cnn],
        [dropout_rate],
        [len(train_dataset)],
        {
            "indices_to_prune_conv1": [indices_to_prune_conv1],
            "indices_to_prune_conv2": [indices_to_prune_conv2],
            "indices_to_prune_conv3": [indices_to_prune_conv3],
            "indices_to_prune_fc": [indices_to_prune_fc],
        },
    )

    # Test the aggregated model
    test_loss_aggregated, accuracy_aggregated, _ = test(
        original_cnn, device, test_loader, criterion
    )
    print(f"Aggregated model test accuracy: {accuracy_aggregated:.6f}")

    if accuracy_aggregated >= accuracy_last_aggregated:
        accuracy_last_aggregated = accuracy_aggregated
    else:
        original_cnn.load_state_dict(original_cnn_backup)
        print("Current aggregation result is worse. Discarding this aggregation.")
    print("*" * 80)
    print("-" * 80)

print(f"Max aggregated model test accuracy: {accuracy_last_aggregated:.6f}")
