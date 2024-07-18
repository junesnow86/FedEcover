import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from modules.heterofl_utils import heterofl_aggregate, prune_cnn
from modules.models import CNN
from modules.utils import (
    test,
    train,
)

ROUNDS = 20
EPOCHS = 1
LR = 0.001
BATCH_SIZE = 128

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=False, transform=transform
)

test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=False, transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

num_train = len(train_dataset)
indices = list(range(num_train))
np.random.shuffle(indices)

num_subsets = 10
subset_size = num_train // num_subsets
subsets_indices = [
    indices[i : i + subset_size] for i in range(0, num_train, subset_size)
]
subset_sizes = [len(subset) for subset in subsets_indices]

dataloaders = [
    DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    for subset_indices in subsets_indices
]

# 示例：如何使用这些数据加载器
# for i, dataloader in enumerate(dataloaders):
#     for images, labels in dataloader:
#         # 在这里处理每个子集的数据
#         pass

global_cnn = CNN()

num_models = 10
num_unpruned = int(num_models * 0.2)
num_pruned = num_models - num_unpruned

unpruned_models = [CNN() for _ in range(num_unpruned)]
for i in range(num_unpruned):
    unpruned_models[i].load_state_dict(global_cnn.state_dict())

p = 0.9

# unpruned_models = [prune_cnn(global_cnn, p) for _ in range(num_unpruned)]
pruned_models = [prune_cnn(global_cnn, p) for _ in range(num_pruned)]
all_client_models = [*unpruned_models, *pruned_models]

# global_cnn = prune_cnn(global_cnn, p)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
for round in range(ROUNDS):
    for i, dataloader in enumerate(dataloaders):
        local_model = all_client_models[i]
        optimizer = optim.Adam(local_model.parameters(), lr=LR)
        train(local_model, device, dataloader, optimizer, criterion, EPOCHS)
        _, local_test_acc, _ = test(local_model, device, test_loader, criterion)
        print(f"Round {round + 1}, Subset {i + 1}, Test Acc: {local_test_acc:.4f}")

    heterofl_aggregate(global_cnn, all_client_models, subset_sizes)

    _, test_acc, _ = test(global_cnn, device, test_loader, criterion)
    print(f"Round {round + 1}, Aggregated Test Acc: {test_acc:.4f}")
    print("=" * 80)
