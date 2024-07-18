import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from modules.heterofl_utils import prune_cnn
from modules.models import CNN
from modules.utils import (
    test,
    train,
    vanilla_federated_averaging,
)

ROUNDS = 50
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

original_cnn = CNN()

p = 0.9
num_models = 10
global_cnn = prune_cnn(original_cnn, p)
all_client_models = [prune_cnn(original_cnn, p) for _ in range(num_models)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
for round in range(ROUNDS):
    for i, dataloader in enumerate(dataloaders):
        local_model = all_client_models[i]
        optimizer = optim.Adam(local_model.parameters(), lr=LR)
        train(local_model, device, dataloader, optimizer, criterion, EPOCHS)
        _, local_test_acc, _ = test(local_model, device, test_loader, criterion)
        print(f"Round {round + 1}, Subset {i + 1}, Test Acc: {local_test_acc:.4f}")

    aggregated_weights = vanilla_federated_averaging(
        models=all_client_models, sample_numbers=subset_sizes
    )
    global_cnn.load_state_dict(aggregated_weights)

    _, test_acc, _ = test(global_cnn, device, test_loader, criterion)
    print(f"Round {round + 1}, Test Acc: {test_acc:.4f}")
    print("=" * 80)
