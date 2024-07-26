import csv
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from modules.heterofl_utils import (
    heterofl_aggregate,
    prune_cnn_group,
)
from modules.models import CNN
from modules.utils import (
    test,
    train,
)

ROUNDS = 100
EPOCHS = 1
LR = 0.001
BATCH_SIZE = 128
NUM_PARTICIPANTS = 10

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


# 首先，获取每个类别的索引
class_indices = {i: [] for i in range(10)}  # CIFAR10有10个类别
for idx, (_, label) in enumerate(train_dataset):
    class_indices[label].append(idx)

split_ratio = 0.8

# 首先，初始化每个参与方的数据索引集合
participant_data_indices = [[] for i in range(NUM_PARTICIPANTS)]

for cls, indices in class_indices.items():
    np.random.shuffle(indices)  # 打乱索引以随机分配
    split_index = int(len(indices) * split_ratio)

    # 为当前类别的主要参与方分配80%的数据
    main_indices = indices[:split_index]
    participant_data_indices[cls].extend(main_indices)

    # 剩余20%的数据均匀分配给其他参与方
    remaining_indices = indices[split_index:]
    num_remaining_per_participant = len(remaining_indices) // (NUM_PARTICIPANTS - 1)

    # 分配剩余数据
    for i, start_index in enumerate(
        range(0, len(remaining_indices), num_remaining_per_participant)
    ):
        if i == NUM_PARTICIPANTS - 1:  # 最后一个参与方获取所有剩余的数据
            participant_data_indices[(cls + i + 1) % NUM_PARTICIPANTS].extend(
                remaining_indices[start_index:]
            )
            break
        participant_data_indices[(cls + i + 1) % NUM_PARTICIPANTS].extend(
            remaining_indices[start_index : start_index + num_remaining_per_participant]
        )

subset_sizes = [len(indices) for indices in participant_data_indices]

# 创建每个参与方的数据加载器
dataloaders = [
    DataLoader(
        Subset(train_dataset, indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    for indices in participant_data_indices
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
results_class = []

for round in range(ROUNDS):
    round_results = {"Round": round + 1}
    round_results_class = {"Round": round + 1}

    # # Load global model's parameters
    # unpruned_models = [CNN() for _ in range(num_unpruned)]
    # for i in range(num_unpruned):
    #     unpruned_models[i].load_state_dict(global_cnn.state_dict())

    # p = 0.9

    # pruned_cnn_groups = []
    # pruned_indices_groups = []
    # for i in range(num_pruned):
    #     pruned_cnn_group, pruned_indices_group = prune_cnn_group(
    #         global_cnn, p, scaling=True
    #     )
    #     pruned_cnn_groups.append(pruned_cnn_group)
    #     pruned_indices_groups.append(pruned_indices_group)

    pruned_cnn_groups = []
    pruned_indices_groups = []
    for i in range(num_models):
        pruned_cnn_group, pruned_indices_group = prune_cnn_group(
            global_cnn, dropout_rates[i], scaling=True
        )
        pruned_cnn_groups.append(pruned_cnn_group)
        pruned_indices_groups.append(pruned_indices_group)

    # Local training
    for i, dataloader in enumerate(dataloaders):
        pruned_cnn_group = pruned_cnn_groups[i]
        acc_group = 0.0
        class_acc_group = {i: 0.0 for i in range(10)}
        for j, pruned_cnn in enumerate(pruned_cnn_group):
            optimizer = optim.Adam(pruned_cnn.parameters(), lr=LR)
            train(pruned_cnn, device, dataloader, optimizer, criterion, EPOCHS)
            _, local_test_acc, local_class_acc = test(
                pruned_cnn, device, test_loader, criterion
            )
            acc_group += local_test_acc
            for k, v in local_class_acc.items():
                class_acc_group[k] += v
        acc_group /= len(pruned_cnn_group)
        for k in class_acc_group:
            class_acc_group[k] /= len(pruned_cnn_group)
        print(
            f"Round {round + 1}, Subset {i + 1}, Group Averaging Test Acc: {acc_group:.4f}\tClass Acc: {local_class_acc}"
        )
        round_results[f"Subset {i + 1}"] = acc_group
        round_results_class[f"Subset {i + 1}"] = class_acc_group
        # if i < num_pruned:
        #     pruned_cnn_group = pruned_cnn_groups[i]
        #     acc_group = 0.0
        #     class_acc_group = {i: 0.0 for i in range(10)}
        #     for j, pruned_cnn in enumerate(pruned_cnn_group):
        #         optimizer = optim.Adam(pruned_cnn.parameters(), lr=LR)
        #         train(pruned_cnn, device, dataloader, optimizer, criterion, EPOCHS)
        #         _, local_test_acc, local_class_acc = test(
        #             pruned_cnn, device, test_loader, criterion
        #         )
        #         # print(
        #         #     f"Round {round + 1}, Subset {i + 1}, Model {j+1}, Test Acc: {local_test_acc:.4f}\tClass Acc: {local_class_acc}"
        #         # )
        #         acc_group += local_test_acc
        #         for k, v in local_class_acc.items():
        #             class_acc_group[k] += v
        #     acc_group /= len(pruned_cnn_group)
        #     for k in class_acc_group:
        #         class_acc_group[k] /= len(pruned_cnn_group)
        #     print(
        #         f"Round {round + 1}, Subset {i + 1}, Group Averaging Test Acc: {acc_group:.4f}\tClass Acc: {class_acc_group}"
        #     )
        #     round_results[f"Subset {i + 1}"] = acc_group
        #     round_results_class[f"Subset {i + 1}"] = class_acc_group
        # else:
        #     local_model = unpruned_models[i - num_pruned]
        #     optimizer = optim.Adam(local_model.parameters(), lr=LR)
        #     train(local_model, device, dataloader, optimizer, criterion, EPOCHS)
        #     _, local_test_acc, local_class_acc = test(
        #         local_model, device, test_loader, criterion
        #     )
        #     print(
        #         f"Round {round + 1}, Subset {i + 1}, Test Acc: {local_test_acc:.4f}\tClass Acc: {local_class_acc}"
        #     )
        #     round_results[f"Subset {i + 1}"] = local_test_acc
        #     round_results_class[f"Subset {i + 1}"] = local_class_acc

    flatten_all_client_models = []
    flatten_subset_sizes = []
    flatten_pruned_indices = []
    # for i in range(num_pruned):
    for i in range(num_models):
        pruned_cnn_group = pruned_cnn_groups[i]
        flatten_all_client_models.extend(pruned_cnn_group)
        flatten_subset_sizes.extend([subset_sizes[i]] * len(pruned_cnn_group))
        flatten_pruned_indices.extend(pruned_indices_groups[i])
    # flatten_all_client_models.extend(unpruned_models)
    # flatten_subset_sizes.extend(subset_sizes[num_pruned:])
    # flatten_pruned_indices.extend([empty_pruned_indices()] * (num_models - num_pruned))

    # Aggregation
    heterofl_aggregate(
        global_cnn,
        flatten_all_client_models,
        flatten_pruned_indices,
        flatten_subset_sizes,
    )

    _, test_acc, class_acc = test(global_cnn, device, test_loader, criterion)
    print(
        f"Round {round + 1}, Aggregated Test Acc: {test_acc:.4f}\tClass Acc: {class_acc}"
    )
    round_results["Aggregated"] = test_acc
    round_results_class["Aggregated"] = class_acc

    print("=" * 80)

    results.append(round_results)
    results_class.append(round_results_class)

with open(
    "results/position_heterofl_more_different_p_unbalanced_classes.csv", "w"
) as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
