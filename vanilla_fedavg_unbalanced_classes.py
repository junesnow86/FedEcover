import csv
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from modules.heterofl_utils import prune_cnn
from modules.models import CNN
from modules.utils import test, train, vanilla_federated_averaging

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

split_ratio = 0.9

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

original_cnn = CNN()

p = 0.9
num_models = 10
global_cnn, _ = prune_cnn(original_cnn, p, position=0)
all_client_models = []
for i in range(num_models):
    client_model, _ = prune_cnn(original_cnn, p, position=0)
    all_client_models.append(client_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

results = []
results_class = []

for round in range(ROUNDS):
    round_results = {"Round": round + 1}
    round_results_class = {"Round": round + 1}

    # Load global model's parameters
    for i in range(num_models):
        all_client_models[i].load_state_dict(global_cnn.state_dict())

    # Local training
    for i, dataloader in enumerate(dataloaders):
        local_model = all_client_models[i]
        optimizer = optim.Adam(local_model.parameters(), lr=LR)
        train(local_model, device, dataloader, optimizer, criterion, EPOCHS)
        _, local_test_acc, local_class_acc = test(
            local_model, device, test_loader, criterion
        )
        print(
            f"Round {round + 1}, Subset {i + 1}, Test Acc: {local_test_acc:.4f}\tClass Acc: {local_class_acc}"
        )
        round_results[f"Subset {i + 1}"] = local_test_acc
        round_results_class[f"Subset {i + 1}"] = local_class_acc

    # Aggregation
    aggregated_weight = vanilla_federated_averaging(
        global_model=global_cnn, models=all_client_models, sample_numbers=subset_sizes
    )
    global_cnn.load_state_dict(aggregated_weight)

    _, test_acc, class_acc = test(global_cnn, device, test_loader, criterion)
    print(
        f"Round {round + 1}, Aggregated Test Acc: {test_acc:.4f}\tClass Acc: {class_acc}"
    )
    round_results["Aggregated"] = test_acc
    round_results_class["Aggregated"] = class_acc
    print("=" * 80)

    results.append(round_results)
    results_class.append(round_results_class)

with open("results/vanilla_fedavg_unbalanced_classes.csv", "w") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

with open("results/vanilla_fedavg_unbalanced_classes.txt", "w") as f:
    f.write(str(results_class))