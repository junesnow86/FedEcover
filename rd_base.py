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
from torchvision.models import resnet18, ResNet

from modules.aggregation import aggregate_cnn, aggregate_resnet18
from modules.models import CNN
from modules.pruning import prune_cnn, prune_resnet18
from modules.utils import (
    test,
    train,
    replace_bn_with_ln,
)

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

ROUNDS = 200
EPOCHS = 1
LR = 0.001
BATCH_SIZE = 128
LR_DECAY = True

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

# global_model = CNN()
global_model = resnet18(weights=None)
replace_bn_with_ln(global_model)

if model_type == "cnn":
    assert isinstance(global_model, CNN), f"Model type should be CNN, but got {type(global_model)}"
elif model_type == "resnet":
    assert isinstance(global_model, ResNet), f"Model type should be ResNet, but got {type(global_model)}"

num_models = 10

# p = 0.8, 0.5, 0.2
dropout_rates = [0.2, 0.2, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 0.8, 0.8]
learning_rates = [LR] * num_models
decay_rounds = [20, 50, 100]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

results_class = []
train_loss_results = []
test_loss_results = []
test_acc_results = []


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

    if model_type == "cnn":
        indices_to_prune_conv1_list = []
        indices_to_prune_conv2_list = []
        indices_to_prune_conv3_list = []
        indices_to_prune_fc_list = []

        for i in range(num_models):
            (
                client_model,
                indices_to_prune_conv1,
                indices_to_prune_conv2,
                indices_to_prune_conv3,
                indices_to_prune_fc,
            ) = prune_cnn(
                global_model,
                dropout_rates[i],
                scaling=True,
            )
            all_client_models.append(client_model)
            indices_to_prune_conv1_list.append(indices_to_prune_conv1)
            indices_to_prune_conv2_list.append(indices_to_prune_conv2)
            indices_to_prune_conv3_list.append(indices_to_prune_conv3)
            indices_to_prune_fc_list.append(indices_to_prune_fc)
    elif model_type == "resnet":
        pruned_indices_dicts = []

        for i in range(num_models):
            client_model, pruned_indices_dict = prune_resnet18(global_model, dropout_rates[i])
            all_client_models.append(client_model)
            pruned_indices_dicts.append(pruned_indices_dict)

    # Local training
    for i, dataloader in enumerate(dataloaders):
        local_model = all_client_models[i]
        if LR_DECAY and dropout_rates[i] == 0.8 and round in decay_rounds:
            learning_rates[i] /= 10
            print(f"Subset {i + 1} learning rate decayed to {learning_rates[i]}")
        lr = learning_rates[i]
        optimizer = optim.Adam(local_model.parameters(), lr=lr)
        local_train_loss = train(
            local_model, optimizer, criterion, dataloader, device=device, epochs=EPOCHS
        )
        local_test_loss, local_test_acc, local_class_acc = test(
            local_model, criterion, test_loader, device=device, num_classes=10
        )
        print(
            f"Subset {i + 1}\tTrain Loss: {local_train_loss:.4f}\tTest Loss: {local_test_loss:.4f}\tTest Acc: {local_test_acc:.4f}\tClass Acc: {local_class_acc}"
        )
        round_train_loss_results[f"Subset {i + 1}"] = local_train_loss
        round_test_loss_results[f"Subset {i + 1}"] = local_test_loss
        round_test_acc_results[f"Subset {i + 1}"] = local_test_acc
        round_results_class[f"Subset {i + 1}"] = local_class_acc

    # Aggregation
    if model_type == "cnn":
        aggregate_cnn(
            global_model,
            all_client_models,
            subset_sizes,
            indices_to_prune_conv1=indices_to_prune_conv1_list,
            indices_to_prune_conv2=indices_to_prune_conv2_list,
            indices_to_prune_conv3=indices_to_prune_conv3_list,
            indices_to_prune_fc=indices_to_prune_fc_list,
        )
    elif model_type == "resnet":
        aggregate_resnet18(
            global_model,
            all_client_models,
            subset_sizes,
            pruned_indices_dicts,
        )

    global_test_loss, global_test_acc, global_class_acc = test(
        global_model, criterion, test_loader, device=device, num_classes=10
    )
    print(
        f"Aggregated Test Loss: {global_test_loss:.4f}\tAggregated Test Acc: {global_test_acc:.4f}\tAggregated Class Acc: {global_class_acc}"
    )
    round_test_loss_results["Aggregated"] = global_test_loss
    round_test_acc_results["Aggregated"] = global_test_acc
    round_results_class["Aggregated"] = global_class_acc

    train_loss_results.append(round_train_loss_results)
    test_loss_results.append(round_test_loss_results)
    test_acc_results.append(round_test_acc_results)
    results_class.append(round_results_class)

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
    with open(os.path.join(save_dir, "rd_base_train_loss.csv"), "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=train_loss_results[0].keys())
        writer.writeheader()
        writer.writerows(train_loss_results)

    with open(os.path.join(save_dir, "rd_base_test_loss.csv"), "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=test_loss_results[0].keys())
        writer.writeheader()
        writer.writerows(test_loss_results)

    with open(os.path.join(save_dir, "rd_base_test_acc.csv"), "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=test_acc_results[0].keys())
        writer.writeheader()
        writer.writerows(test_acc_results)

    print("Results saved.")
else:
    print("Results not saved.")
