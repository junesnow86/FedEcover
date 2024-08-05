import csv
import random
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from modules.aggregation import aggregate_cnn
from modules.models import CNN
from modules.pruning import prune_cnn_into_groups
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
num_unpruned = int(num_models * 0.2)
num_pruned = num_models - num_unpruned

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

results = []

start_time = time()

for round in range(ROUNDS):
    round_results = {"Round": round + 1}

    p = 0.9
    flatten_all_models = []
    flatten_all_subset_sizes = []
    flatten_indices_to_prune_conv1_list = []
    flatten_indices_to_prune_conv2_list = []
    flatten_indices_to_prune_conv3_list = []
    flatten_indices_to_prune_fc_list = []

    for i in range(num_pruned):
        group_acc = 0.0
        group_pruned_models, group_indices_to_prune = prune_cnn_into_groups(
            global_cnn,
            p,
            scaling=True,
        )
        group_indices_to_prune_conv1 = [
            indices_to_prune["indices_to_prune_conv1"]
            for indices_to_prune in group_indices_to_prune
        ]
        group_indices_to_prune_conv2 = [
            indices_to_prune["indices_to_prune_conv2"]
            for indices_to_prune in group_indices_to_prune
        ]
        group_indices_to_prune_conv3 = [
            indices_to_prune["indices_to_prune_conv3"]
            for indices_to_prune in group_indices_to_prune
        ]
        group_indices_to_prune_fc = [
            indices_to_prune["indices_to_prune_fc"]
            for indices_to_prune in group_indices_to_prune
        ]
        dataloader = dataloaders[i]
        for pruned_model in group_pruned_models:
            optimizer = optim.Adam(pruned_model.parameters(), lr=LR)
            train(pruned_model, device, dataloader, optimizer, criterion, EPOCHS)
            _, local_test_acc, _ = test(pruned_model, device, dataloader, criterion)
            group_acc += local_test_acc
        group_acc /= len(group_pruned_models)
        print(
            f"Round {round + 1}, Subset {i + 1}, Group Averaging Test Acc: {group_acc:.4f}"
        )
        round_results[f"Subset {i + 1}"] = group_acc
        flatten_all_models.extend(group_pruned_models)
        flatten_all_subset_sizes.extend([subset_sizes[i]] * len(group_pruned_models))
        flatten_indices_to_prune_conv1_list.extend(group_indices_to_prune_conv1)
        flatten_indices_to_prune_conv2_list.extend(group_indices_to_prune_conv2)
        flatten_indices_to_prune_conv3_list.extend(group_indices_to_prune_conv3)
        flatten_indices_to_prune_fc_list.extend(group_indices_to_prune_fc)

    for i in range(num_pruned, num_models):
        unpruned_model = CNN()
        unpruned_model.load_state_dict(global_cnn.state_dict())
        dataloader = dataloaders[i]
        optimizer = optim.Adam(unpruned_model.parameters(), lr=LR)
        train(unpruned_model, device, dataloader, optimizer, criterion, EPOCHS)
        _, local_test_acc, _ = test(unpruned_model, device, dataloader, criterion)
        print(f"Round {round + 1}, Subset {i + 1}, Test Acc: {local_test_acc:.4f}")
        round_results[f"Subset {i + 1}"] = local_test_acc
        flatten_all_models.append(unpruned_model)
        flatten_all_subset_sizes.append(subset_sizes[i])
        flatten_indices_to_prune_conv1_list.append({})
        flatten_indices_to_prune_conv2_list.append({})
        flatten_indices_to_prune_conv3_list.append({})
        flatten_indices_to_prune_fc_list.append({})

    # Aggregation
    print(f"Aggregating {len(flatten_all_models)} models...")
    aggregate_cnn(
        global_cnn,
        flatten_all_models,
        flatten_all_subset_sizes,
        indices_to_prune_conv1=flatten_indices_to_prune_conv1_list,
        indices_to_prune_conv2=flatten_indices_to_prune_conv2_list,
        indices_to_prune_conv3=flatten_indices_to_prune_conv3_list,
        indices_to_prune_fc=flatten_indices_to_prune_fc_list,
    )

    _, test_acc, _ = test(global_cnn, device, test_loader, criterion)
    print(f"Round {round + 1}, Aggregated Test Acc: {test_acc:.4f}")
    round_results["Aggregated"] = test_acc
    print("=" * 80)

    results.append(round_results)

end_time = time()
print(f"All rounds time cost: {end_time - start_time:.2f}s")

with open("results/group_random_dropout.csv", "w") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
