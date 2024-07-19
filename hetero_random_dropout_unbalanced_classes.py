import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from modules.models import CNN
from modules.utils import (
    aggregate_cnn,
    prune_cnn,
    test,
    train,
)

ROUNDS = 20
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


# 首先，获取每个类别的索引
class_indices = {i: [] for i in range(10)}  # CIFAR10有10个类别
for idx, (_, label) in enumerate(train_dataset):
    class_indices[label].append(idx)

# 按类别划分成两个大的子集，一个包含80%的类别，另一个包含20%的类别
num_classes = len(class_indices)
split_ratio = 0.8
split_index = int(num_classes * split_ratio)

# 类别索引顺序集
classes_80_percent = list(range(split_index))  # 80%的类别
classes_20_percent = list(range(split_index, num_classes))  # 剩下20%的类别

# 获取对应类别的索引
indices_80_percent = [
    index for cls in classes_80_percent for index in class_indices[cls]
]
indices_20_percent = [
    index for cls in classes_20_percent for index in class_indices[cls]
]

# 将80%类别的子集均匀划分成8个子集
np.random.shuffle(indices_80_percent)
num_subsets_80 = 8
subset_size_80 = len(indices_80_percent) // num_subsets_80
subsets_indices_80 = [
    indices_80_percent[i : i + subset_size_80]
    for i in range(0, len(indices_80_percent), subset_size_80)
]
subset_sizes_80 = [len(subset) for subset in subsets_indices_80]

# 将20%类别的子集均匀划分成2个子集
np.random.shuffle(indices_20_percent)
num_subsets_20 = 2
subset_size_20 = len(indices_20_percent) // num_subsets_20
subsets_indices_20 = [
    indices_20_percent[i : i + subset_size_20]
    for i in range(0, len(indices_20_percent), subset_size_20)
]
subset_sizes_20 = [len(subset) for subset in subsets_indices_20]

# 创建数据加载器，每个加载器对应一个数据子集
dataloaders_80 = [
    DataLoader(
        Subset(train_dataset, subset_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    for subset_indices in subsets_indices_80
]

dataloaders_20 = [
    DataLoader(
        Subset(train_dataset, subset_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    for subset_indices in subsets_indices_20
]

dataloaders = dataloaders_80 + dataloaders_20
subset_sizes = subset_sizes_80 + subset_sizes_20

# num_train = len(train_dataset)
# indices = list(range(num_train))
# np.random.shuffle(indices)

# num_subsets = 10
# subset_size = num_train // num_subsets
# subsets_indices = [
#     indices[i : i + subset_size] for i in range(0, num_train, subset_size)
# ]
# subset_sizes = [len(subset) for subset in subsets_indices]

# dataloaders = [
#     DataLoader(
#         Subset(train_dataset, subset_indices),
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#     )
#     for subset_indices in subsets_indices
# ]

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

for round in range(ROUNDS):
    # Load global model's parameters
    unpruned_models = [CNN() for _ in range(num_unpruned)]
    for i in range(num_unpruned):
        unpruned_models[i].load_state_dict(global_cnn.state_dict())

    p = 0.9

    # pruned_models = [prune_cnn(global_cnn, p) for _ in range(num_pruned)]
    pruned_models = []
    indices_to_prune_conv1_list = []
    indices_to_prune_conv2_list = []
    indices_to_prune_conv3_list = []
    indices_to_prune_fc_list = []
    for _ in range(num_pruned):
        (
            pruned_model,
            indices_to_prune_conv1,
            indices_to_prune_conv2,
            indices_to_prune_conv3,
            indices_to_prune_fc,
        ) = prune_cnn(
            global_cnn,
            p,
            scaling=True,
        )
        pruned_models.append(pruned_model)
        indices_to_prune_conv1_list.append(indices_to_prune_conv1)
        indices_to_prune_conv2_list.append(indices_to_prune_conv2)
        indices_to_prune_conv3_list.append(indices_to_prune_conv3)
        indices_to_prune_fc_list.append(indices_to_prune_fc)
    for _ in range(num_unpruned):
        indices_to_prune_conv1_list.append({})
        indices_to_prune_conv2_list.append({})
        indices_to_prune_conv3_list.append({})
        indices_to_prune_fc_list.append({})
    all_client_models = [*pruned_models, *unpruned_models]

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

    # Aggregation
    aggregate_cnn(
        global_cnn,
        all_client_models,
        subset_sizes,
        indices_to_prune_conv1=indices_to_prune_conv1_list,
        indices_to_prune_conv2=indices_to_prune_conv2_list,
        indices_to_prune_conv3=indices_to_prune_conv3_list,
        indices_to_prune_fc=indices_to_prune_fc_list,
    )

    _, test_acc, class_acc = test(global_cnn, device, test_loader, criterion)
    print(
        f"Round {round + 1}, Aggregated Test Acc: {test_acc:.4f}\tClass Acc: {class_acc}"
    )
    print("=" * 80)
