import csv
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from modules.heterofl_utils import heterofl_aggregate, prune_cnn
from modules.models import CNN
from modules.utils import (
    test,
    train,
)

ROUNDS = 100
EPOCHS = 1
LR = 0.001
BATCH_SIZE = 128

seed = 18
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
        Subset(train_dataset, subset_indices),
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
# num_unpruned = int(num_models * 0.2)
# num_pruned = num_models - num_unpruned

# p = 0.8, 0.5, 0.2
dropout_rates = [0.2, 0.2, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 0.8, 0.8]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

results = []

for round in range(ROUNDS):
    round_results = {"Round": round + 1}

    # # Load global model's parameters
    # unpruned_models = [CNN() for _ in range(num_unpruned)]
    # for i in range(num_unpruned):
    #     unpruned_models[i].load_state_dict(global_cnn.state_dict())

    # p = 0.9

    # pruned_models = [prune_cnn(global_cnn, p, position=0) for _ in range(num_pruned)]
    # pruned_models = []
    # pruned_indices_list = []
    # for i in range(num_pruned):
    #     pruned_model, pruned_indices = prune_cnn(global_cnn, p, position=0)
    #     pruned_models.append(pruned_model)
    #     pruned_indices_list.append(pruned_indices)
    # all_client_models = [*pruned_models, *unpruned_models]
    # pruned_indices_list.extend([empty_pruned_indices()] * num_unpruned)

    all_client_models = []
    pruned_indices_list = []
    for i in range(num_models):
        p = dropout_rates[i]
        client_model, pruned_indices = prune_cnn(global_cnn, p, position=0)
        all_client_models.append(client_model)
        pruned_indices_list.append(pruned_indices)

    # Local training
    for i, dataloader in enumerate(dataloaders):
        local_model = all_client_models[i]
        optimizer = optim.Adam(local_model.parameters(), lr=LR)
        train(local_model, device, dataloader, optimizer, criterion, EPOCHS)
        _, local_test_acc, _ = test(local_model, device, test_loader, criterion)
        print(f"Round {round + 1}, Subset {i + 1}, Test Acc: {local_test_acc:.4f}")
        round_results[f"Subset {i + 1}"] = local_test_acc

    # Aggregation
    heterofl_aggregate(
        global_cnn, all_client_models, pruned_indices_list, subset_sizes
    )  # For convenient pruning
    pruned_global_cnn, _ = prune_cnn(
        global_cnn, 0.2, position=0
    )  # Use the largest client model size as the global model size
    heterofl_aggregate(
        pruned_global_cnn, all_client_models, pruned_indices_list, subset_sizes
    )

    _, test_acc, _ = test(pruned_global_cnn, device, test_loader, criterion)
    print(f"Round {round + 1}, Pruned-global Aggregated Test Acc: {test_acc:.4f}")
    round_results["Pruned-global Aggregated"] = test_acc

    _, test_acc, _ = test(global_cnn, device, test_loader, criterion)
    print(f"Round {round + 1}, Whole Aggregated Test Acc: {test_acc:.4f}")
    round_results["Whole Aggregated"] = test_acc

    print("=" * 80)

    results.append(round_results)

with open("results/heterofl_more_different_p.csv", "w") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
