import csv
import os
import random
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18

from modules.aggregation import vanilla_federated_averaging, aggregate_resnet18_vanilla
from modules.heterofl_utils import prune_cnn
from modules.models import CNN
from modules.pruning import prune_resnet18
from modules.utils import replace_bn_with_ln, test, train, calculate_model_size
from modules.debugging import create_empty_pruned_indices_dict
from modules.args_parser import get_args

args = get_args()
save_dir = args.save_dir
model_type = args.model

ROUNDS = args.round
EPOCHS = args.epochs
LR = args.lr
BATCH_SIZE = args.batch_size

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

if model_type == "cnn":
    original_cnn = CNN()
elif model_type == "resnet":
    original_cnn = resnet18(weights=None)
    replace_bn_with_ln(original_cnn)
else:
    raise ValueError(f"Model type {model_type} not supported.")

p = 0.8
if model_type == "cnn":
    global_cnn, _ = prune_cnn(original_cnn, p, position=0)
elif model_type == "resnet":
    global_cnn, _ = prune_resnet18(original_cnn, p)
else:
    raise ValueError(f"Model type {model_type} not supported.")

num_models = 10
all_client_models = []
for i in range(num_models):
    if model_type == "cnn":
        client_model, _ = prune_cnn(original_cnn, p, position=0)
    elif model_type == "resnet":
        client_model, _ = prune_resnet18(original_cnn, p)
    else:
        raise ValueError(f"Model type {model_type} not supported.")
    all_client_models.append(client_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

train_loss_results = []
test_loss_results = []
test_acc_results = []
class_acc_results = []

start_time = time()
for round in range(ROUNDS):
    print(f"Round {round + 1}")
    round_start_time = time()

    round_train_loss_results = {"Round": round + 1}
    round_test_loss_results = {"Round": round + 1}
    round_test_acc_results = {"Round": round + 1}
    round_class_acc_results = {"Round": round + 1}

    # Load global model's parameters
    for i in range(num_models):
        all_client_models[i].load_state_dict(global_cnn.state_dict())

    # Local training
    for i, dataloader in enumerate(dataloaders):
        local_model = all_client_models[i]
        optimizer = optim.Adam(local_model.parameters(), lr=LR)
        local_train_loss = train(
            local_model, optimizer, criterion, dataloader, device, EPOCHS
        )
        local_test_loss, local_test_acc, local_class_acc = test(
            local_model, criterion, test_loader, device, num_classes=10
        )
        model_size = calculate_model_size(local_model, print_result=False, unit="MB")
        print(
            f"Subset {i + 1}\tModel Size: {model_size:.2f} MB\tTrain Loss: {local_train_loss:.4f}\tTest Loss: {local_test_loss:.4f}\tTest Acc: {local_test_acc:.4f}\tClass Acc: {local_class_acc}"
        )
        round_train_loss_results[f"Subset {i + 1}"] = local_train_loss
        round_test_loss_results[f"Subset {i + 1}"] = local_test_loss
        round_test_acc_results[f"Subset {i + 1}"] = local_test_acc
        round_class_acc_results[f"Subset {i + 1}"] = local_class_acc

    # Aggregation
    if model_type == "cnn":
        aggregated_weight = vanilla_federated_averaging(
            models=all_client_models, sample_numbers=subset_sizes
        )
        global_cnn.load_state_dict(aggregated_weight)
    elif model_type == "resnet":
        aggregate_resnet18_vanilla(
            global_model=global_cnn,
            local_models=all_client_models,
            client_weights=subset_sizes,
            pruned_indices_dicts=[create_empty_pruned_indices_dict() for _ in range(num_models)],
        )
    else:
        raise ValueError(f"Model type {model_type} not supported.")

    global_test_loss, global_test_acc, global_class_acc = test(
        global_cnn, criterion, test_loader, device, num_classes=10
    )
    print(
        f"Aggregated Test Loss: {global_test_loss:.4f}\tAggregated Test Acc: {global_test_acc:.4f}\tAggregated Class Acc: {global_class_acc}"
    )
    round_test_loss_results["Aggregated"] = global_test_loss
    round_test_acc_results["Aggregated"] = global_test_acc
    round_class_acc_results["Aggregated"] = global_class_acc

    train_loss_results.append(round_train_loss_results)
    test_loss_results.append(round_test_loss_results)
    test_acc_results.append(round_test_acc_results)
    class_acc_results.append(round_class_acc_results)

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

if save_dir is not None:
    with open(
        os.path.join(save_dir, f"vanilla_{model_type}_train_loss.csv"), "w"
    ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=train_loss_results[0].keys())
        writer.writeheader()
        writer.writerows(train_loss_results)

    with open(
        os.path.join(save_dir, f"vanilla_{model_type}_test_loss.csv"), "w"
    ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=test_loss_results[0].keys())
        writer.writeheader()
        writer.writerows(test_loss_results)

    with open(
        os.path.join(save_dir, f"vanilla_{model_type}_test_acc.csv"), "w"
    ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=test_acc_results[0].keys())
        writer.writeheader()
        writer.writerows(test_acc_results)

    print("Results saved.")
else:
    print("Results not saved.")
