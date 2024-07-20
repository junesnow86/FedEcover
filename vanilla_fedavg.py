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

# 将训练数据集均匀划分成10个子集
num_subsets = 10
subset_size = num_train // num_subsets
subsets_indices = [
    indices[i : i + subset_size] for i in range(0, num_train, subset_size)
]
subset_sizes = [len(subset) for subset in subsets_indices]

# 创建10个数据加载器，每个加载器对应一个数据子集
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

for round in range(ROUNDS):
    round_results = {"Round": round + 1}

    # Load global model's parameters
    for i in range(num_models):
        all_client_models[i].load_state_dict(global_cnn.state_dict())

    # Local training
    for i, dataloader in enumerate(dataloaders):
        local_model = all_client_models[i]
        optimizer = optim.Adam(local_model.parameters(), lr=LR)
        train(local_model, device, dataloader, optimizer, criterion, EPOCHS)
        _, local_test_acc, _ = test(local_model, device, test_loader, criterion)
        print(f"Round {round + 1}, Subset {i + 1}, Test Acc: {local_test_acc:.4f}")
        round_results[f"Subset {i + 1}"] = local_test_acc

    # Aggregation
    aggregated_weight = vanilla_federated_averaging(
        global_model=global_cnn, models=all_client_models, sample_numbers=subset_sizes
    )
    global_cnn.load_state_dict(aggregated_weight)

    _, test_acc, _ = test(global_cnn, device, test_loader, criterion)
    print(f"Round {round + 1}, Aggregated Test Acc: {test_acc:.4f}")
    round_results["Aggregated"] = test_acc
    print("=" * 80)

    results.append(round_results)

with open("results/vanilla_fedavg.csv", "w") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
