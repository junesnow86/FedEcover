import argparse
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
from modules.pruning import prune_cnn
from modules.utils import (
    test,
    train,
)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Random Dropout Scale Small Models")
parser.add_argument(
    "--scale-mode",
    type=str,
    choices=["1x", "2x", "square"],
    default="square",
    help="Scaling mode for the number of models",
)
args = parser.parse_args()
scale_mode = args.scale_mode
print(f"Scale mode: {scale_mode}")
if scale_mode == "1x":
    scale_factor = 1
elif scale_mode == "2x":
    scale_factor = 2

ROUNDS = 200
EPOCHS = 1
LR = 0.001
BATCH_SIZE = 128
NUM_PARTICIPANTS = 10

# Set random seed for reproducibility
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

global_cnn = CNN()

num_models = 10

# p = 0.8, 0.5, 0.2
dropout_rates = [0.2, 0.2, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 0.8, 0.8]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

train_loss_results = []
test_loss_results = []
test_acc_results = []
results_class = []

# Training by rounds
start_time = time()
for round in range(ROUNDS):
    print(f"Round {round + 1}")
    round_start_time = time()

    round_train_loss_results = {"Round": round + 1}
    round_test_loss_results = {"Round": round + 1}
    round_test_acc_results = {"Round": round + 1}
    round_results_class = {"Round": round + 1}

    all_client_models = []
    all_client_model_groups = []
    indices_to_prune_conv1_list = []
    indices_to_prune_conv2_list = []
    indices_to_prune_conv3_list = []
    indices_to_prune_fc_list = []
    flatten_subset_sizes = []

    for i in range(num_models):
        dropout_rate = dropout_rates[i]
        num_models_current_dropout_rate = int(1 / (1 - dropout_rate))
        if scale_mode == "1x" or scale_mode == "2x":
            num_models_current_dropout_rate *= scale_factor
        elif scale_mode == "square":
            num_models_current_dropout_rate **= 2
        print(
            f"Subset {i + 1}\tDropout rate: {dropout_rate}\tNumber of models: {num_models_current_dropout_rate}"
        )
        client_model_group = []
        for _ in range(num_models_current_dropout_rate):
            (
                client_model,
                indices_to_prune_conv1,
                indices_to_prune_conv2,
                indices_to_prune_conv3,
                indices_to_prune_fc,
            ) = prune_cnn(
                global_cnn,
                dropout_rates[i],
                scaling=True,
            )
            client_model_group.append(client_model)
            indices_to_prune_conv1_list.append(indices_to_prune_conv1)
            indices_to_prune_conv2_list.append(indices_to_prune_conv2)
            indices_to_prune_conv3_list.append(indices_to_prune_conv3)
            indices_to_prune_fc_list.append(indices_to_prune_fc)
        all_client_model_groups.append(client_model_group)
        flatten_subset_sizes.extend([subset_sizes[i]] * num_models_current_dropout_rate)
    # Flatten the client model groups
    all_client_models = [
        client_model
        for client_model_group in all_client_model_groups
        for client_model in client_model_group
    ]

    # Local training
    for i, dataloader in enumerate(dataloaders):
        avg_local_train_loss = 0.0
        avg_local_test_loss = 0.0
        avg_local_test_acc = 0.0
        avg_local_class_acc = {}
        for j, local_model in enumerate(all_client_model_groups[i]):
            optimizer = optim.Adam(local_model.parameters(), lr=LR)
            local_train_loss = train(
                local_model,
                optimizer,
                criterion,
                dataloader,
                device=device,
                epochs=EPOCHS,
            )
            local_test_loss, local_test_acc, local_class_acc = test(
                local_model, criterion, test_loader, device=device, num_classes=10
            )
            avg_local_train_loss += local_train_loss
            avg_local_test_loss += local_test_loss
            avg_local_test_acc += local_test_acc
            for cls, acc in local_class_acc.items():
                avg_local_class_acc[cls] = avg_local_class_acc.get(cls, 0.0) + acc
        avg_local_train_loss /= len(all_client_model_groups[i])
        avg_local_test_loss /= len(all_client_model_groups[i])
        avg_local_test_acc /= len(all_client_model_groups[i])
        for cls in avg_local_class_acc:
            avg_local_class_acc[cls] /= len(all_client_model_groups[i])
        print(
            f"Subset {i + 1}\tTrain Loss: {local_train_loss:.4f}\tTest Loss: {local_test_loss:.4f}\tTest Acc: {local_test_acc:.4f}\tClass Acc: {local_class_acc}"
        )
        round_train_loss_results[f"Subset {i + 1}"] = avg_local_train_loss
        round_test_loss_results[f"Subset {i + 1}"] = avg_local_test_loss
        round_test_acc_results[f"Subset {i + 1}"] = avg_local_test_acc
        round_results_class[f"Subset {i + 1}"] = avg_local_class_acc

    # Aggregation
    aggregate_cnn(
        global_cnn,
        all_client_models,
        flatten_subset_sizes,
        indices_to_prune_conv1=indices_to_prune_conv1_list,
        indices_to_prune_conv2=indices_to_prune_conv2_list,
        indices_to_prune_conv3=indices_to_prune_conv3_list,
        indices_to_prune_fc=indices_to_prune_fc_list,
    )

    global_test_loss, global_test_acc, global_class_acc = test(
        global_cnn, criterion, test_loader, device=device, num_classes=10
    )
    print(
        f"Aggregated Test Loss: {global_test_loss:.4f}\tAggregated Test Acc: {global_test_acc:.4f}\tAggregated Class Acc: {global_class_acc}"
    )

    train_loss_results.append(round_train_loss_results)
    test_loss_results.append(round_test_loss_results)
    test_acc_results.append(round_test_acc_results)
    round_results_class["Aggregated"] = global_class_acc

    round_end_time = time()
    round_use_time = round_end_time - round_start_time
    print(
        f"Round {round + 1} use time: {round_use_time/60:.2f} min, ETA: {(ROUNDS - round - 1) * round_use_time / 3600:.2f} hours"
    )
    print("=" * 80)

end_time = time()
print(f"Total use time: {(end_time - start_time) / 3600:.2f} hours")

# Save results to files
def format_results(results):
    formatted_results = []
    for result in results:
        formatted_result = {
            key: f"{value:.4f}" if isinstance(value, float) else value
            for key, value in result.items()
        }
        formatted_results.append(formatted_result)
    return formatted_results


train_loss_results = format_results(train_loss_results)
test_loss_results = format_results(test_loss_results)
test_acc_results = format_results(test_acc_results)

with open(f"results_0815/rd_{scale_mode}_unbalanced_train_loss.csv", "w") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=train_loss_results[0].keys())
    writer.writeheader()
    writer.writerows(train_loss_results)

with open(f"results_0815/rd_{scale_mode}_unbalanced_test_loss.csv", "w") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=test_loss_results[0].keys())
    writer.writeheader()
    writer.writerows(test_loss_results)

with open(f"results_0815/rd_{scale_mode}_unbalanced_test_acc.csv", "w") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=test_acc_results[0].keys())
    writer.writeheader()
    writer.writerows(test_acc_results)
