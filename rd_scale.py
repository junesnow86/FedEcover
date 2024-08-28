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

from modules.aggregation import aggregate_cnn
from modules.models import CNN
from modules.pruning import prune_cnn
from modules.utils import (
    test,
    train,
)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--scale-mode",
    type=str,
    choices=["1x", "2x", "square"],
    default="1x",
    help="Scaling mode for the number of models",
)
parser.add_argument(
    "--save-dir",
    type=str,
    default=None,
    help="Directory to save the results",
)
args = parser.parse_args()
scale_mode = args.scale_mode
print(f"Scale mode: {scale_mode}")
if scale_mode == "1x":
    scale_factor = 1
elif scale_mode == "2x":
    scale_factor = 2
save_dir = args.save_dir
print(f"Save directory: {save_dir}")

ROUNDS = 200
EPOCHS = 1
LR = 0.001
BATCH_SIZE = 128
LR_DECAY = False

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

global_cnn = CNN()

num_models = 10

# p = 0.8, 0.5, 0.2
dropout_rates = [0.2, 0.2, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 0.8, 0.8]
learning_rates = [LR] * num_models
decay_rounds = [20, 50, 100]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

class_acc_results = []
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
    round_class_acc_results = {"Round": round + 1}

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
        if LR_DECAY and dropout_rates[i] == 0.8 and round in decay_rounds:
            learning_rates[i] /= 10
            print(f"Subset {i + 1} learning rate decayed to {learning_rates[i]}")
        lr = learning_rates[i]

        avg_local_train_loss = 0.0
        avg_local_test_loss = 0.0
        avg_local_test_acc = 0.0
        avg_local_class_acc = {}

        for j, local_model in enumerate(all_client_model_groups[i]):
            optimizer = optim.Adam(local_model.parameters(), lr=lr)
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
            f"Subset {i + 1}\tTrain Loss: {avg_local_train_loss:.4f}\tTest Loss: {avg_local_test_loss:.4f}\tTest Acc: {avg_local_test_acc:.4f}\tClass Acc: {avg_local_class_acc}"
        )
        round_train_loss_results[f"Subset {i + 1}"] = avg_local_train_loss
        round_test_loss_results[f"Subset {i + 1}"] = avg_local_test_loss
        round_test_acc_results[f"Subset {i + 1}"] = avg_local_test_acc
        round_class_acc_results[f"Subset {i + 1}"] = avg_local_class_acc

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
    with open(os.path.join(save_dir, f"rd_{scale_mode}_train_loss.csv"), "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=train_loss_results[0].keys())
        writer.writeheader()
        writer.writerows(train_loss_results)

    with open(os.path.join(save_dir, f"rd_{scale_mode}_test_loss.csv"), "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=test_loss_results[0].keys())
        writer.writeheader()
        writer.writerows(test_loss_results)

    with open(os.path.join(save_dir, f"rd_{scale_mode}_test_acc.csv"), "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=test_acc_results[0].keys())
        writer.writeheader()
        writer.writerows(test_acc_results)

    print("Results saved.")
else:
    print("Results not saved.")
