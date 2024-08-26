import argparse
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
from modules.utils import replace_bn_with_ln, test, train
from modules.debugging import create_empty_pruned_indices_dict

parser = argparse.ArgumentParser()
parser.add_argument(
    "--save-dir",
    type=str,
    default=None,
    help="Directory to save the results",
)
parser.add_argument(
    "--model",
    type=str,
    choices=["cnn", "resnet"],
    default="cnn",
    help="Model to use for training",
)
args = parser.parse_args()
save_dir = args.save_dir
model_type = args.model
print(f"Model type: {model_type}")
print(f"Save directory: {save_dir}")
if save_dir is None:
    print("Results will not be saved.")

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

# original_cnn = CNN()
original_cnn = resnet18(weights=None)
replace_bn_with_ln(original_cnn)

p = 0.8
num_models = 10
# global_cnn, _ = prune_cnn(original_cnn, p, position=0)
global_cnn, _ = prune_resnet18(original_cnn, p)
all_client_models = []
for i in range(num_models):
    # client_model, _ = prune_cnn(original_cnn, p, position=0)
    client_model, _ = prune_resnet18(original_cnn, p)
    all_client_models.append(client_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

train_loss_results = []
test_loss_results = []
test_acc_results = []
class_acc_results = []

start_time = time()
for round in range(ROUNDS):
    round_results = {"Round": round + 1}
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
        print(
            f"Subset {i + 1}\tTrain Loss: {local_train_loss:.4f}\tTest Loss: {local_test_loss:.4f}\tTest Acc: {local_test_acc:.4f}\tClass Acc: {local_class_acc}"
        )
        round_train_loss_results[f"Subset {i + 1}"] = local_train_loss
        round_test_loss_results[f"Subset {i + 1}"] = local_test_loss
        round_test_acc_results[f"Subset {i + 1}"] = local_test_acc
        round_class_acc_results[f"Subset {i + 1}"] = local_class_acc

    # Aggregation
    # aggregated_weight = vanilla_federated_averaging(
    #     models=all_client_models, sample_numbers=subset_sizes
    # )
    # global_cnn.load_state_dict(aggregated_weight)
    aggregate_resnet18_vanilla(
        global_model=global_cnn,
        local_models=all_client_models,
        client_weights=subset_sizes,
        pruned_indices_dicts=[create_empty_pruned_indices_dict() for _ in range(num_models)],
    )

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
