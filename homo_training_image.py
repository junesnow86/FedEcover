import copy
import csv
import os
import pickle
import random
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18

from modules.aggregation import vanilla_federated_averaging
from modules.args_parser import get_args
from modules.constants import NORMALIZATION_STATS
from modules.data import create_non_iid_data
from modules.evaluation import evaluate_acc
from modules.models import CNN
from modules.pruning import prune_cnn, prune_resnet18
from modules.training import train
from modules.utils import calculate_model_size, replace_bn_with_ln

# <======================================== Parse arguments ========================================>
args = get_args()
SAVE_DIR = args.save_dir
MODEL_TYPE = args.model
DATASET = args.dataset
if DATASET == "cifar10":
    NUM_CLASSES = 10
elif DATASET == "cifar100":
    NUM_CLASSES = 100
else:
    raise ValueError(f"Dataset {DATASET} not supported.")
ROUNDS = args.rounds
EPOCHS = args.epochs
LR = args.lr
BATCH_SIZE = args.batch_size
NUM_CLIENTS = args.num_clients
AGG_WAY = args.aggregation
DEBUGGING = args.debugging
METHOD = args.method
SELECT_RATIO = args.select_ratio
LOCAL_VALIDATION_FREQUENCY = int(1 / args.select_ratio)
LOCAL_TRAIN_RATIO = args.local_train_ratio

assert METHOD in ["fedavg"]

# Set random seed for reproducibility
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# <======================================== Data preparation ========================================>
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            NORMALIZATION_STATS[DATASET]["mean"], NORMALIZATION_STATS[DATASET]["std"]
        ),
    ]
)

if DATASET == "cifar10":
    global_train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=False, transform=transform
    )

    global_test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=False, transform=transform
    )
elif DATASET == "cifar100":
    global_train_dataset = datasets.CIFAR100(
        root="./data", train=True, download=False, transform=transform
    )

    global_test_dataset = datasets.CIFAR100(
        root="./data", train=False, download=False, transform=transform
    )
else:
    raise ValueError(f"Dataset {DATASET} not supported.")

global_test_loader = DataLoader(
    global_test_dataset, batch_size=BATCH_SIZE, shuffle=False
)

if args.distribution == "iid":
    num_subsets = NUM_CLIENTS
    num_train = len(global_train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    subset_size = num_train // num_subsets
    subset_indices_list = [
        indices[i : i + subset_size] for i in range(0, num_train, subset_size)
    ]
    num_remain = num_train % num_subsets
    if num_remain > 0:
        remaining_indices = indices[-num_remain:]
        for i, index in enumerate(remaining_indices):
            subset_indices_list[i].append(index)
    subset_sizes = [len(indices_a_subset) for indices_a_subset in subset_indices_list]
elif args.distribution == "non-iid":
    num_clients = NUM_CLIENTS
    alpha = args.alpha
    subset_indices_list, client_stats = create_non_iid_data(
        global_train_dataset, num_clients, alpha, seed=seed
    )
    subset_sizes = [len(indices_a_subset) for indices_a_subset in subset_indices_list]

    if args.plot_data_distribution:
        client_ids = list(range(num_clients))
        total_samples = [client_stats[i]["total_samples"] for i in client_ids]
        class_distributions = [
            client_stats[i]["class_distribution"] for i in client_ids
        ]

        # Plot the total number of samples of each client
        plt.figure(figsize=(10, 5))
        plt.bar(client_ids, total_samples, color="b", alpha=0.7)
        plt.title("Total Number of Samples of Each Client")
        plt.xlabel("Client ID")
        plt.ylabel("Number of Samples")
        plt.savefig(f"num_samples_distribution_{DATASET}_alpha{alpha}.png")

        # Plot the class distribution of each client
        num_classes = len(class_distributions[0])
        ind = np.arange(num_classes)
        width = 0.35
        colors = plt.get_cmap("tab20", len(client_ids))
        bottom = np.zeros(num_classes)
        plt.figure(figsize=(20, 10))
        for i, client_id in enumerate(client_ids):
            class_counts = list(class_distributions[i].values())
            plt.bar(
                ind,
                class_counts,
                width,
                label=f"Client {client_id}",
                color=colors(i),
                bottom=bottom,
                alpha=0.7,
            )
            bottom += class_counts
        plt.title("Class Distribution of Each Client")
        plt.xlabel("Class")
        plt.ylabel("Number of Samples")
        plt.xticks(ind, [str(i) for i in range(NUM_CLASSES)])
        plt.legend(bbox_to_anchor=(0.5, -0.1), loc="upper center", ncol=10)
        plt.savefig(
            f"class_distribution_{DATASET}_alpha{alpha}.png", bbox_inches="tight"
        )
else:
    raise ValueError(f"Data distribution {args.distribution} not supported.")

# Split every client's data into train and validation sets
train_ratio = LOCAL_TRAIN_RATIO
train_loaders = []
val_loaders = []
for indices_a_subset in subset_indices_list:
    split_point = int(train_ratio * len(indices_a_subset))
    np.random.shuffle(indices_a_subset)
    train_indices = indices_a_subset[:split_point]
    val_indices = indices_a_subset[split_point:]
    # print(len(train_indices), len(val_indices))
    train_loaders.append(
        DataLoader(
            Subset(global_train_dataset, train_indices),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
    )
    val_loaders.append(
        DataLoader(
            Subset(global_train_dataset, val_indices),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
    )


# <======================================== Model preparation ========================================>
capacity = 0.1
if MODEL_TYPE == "cnn":
    global_model = CNN(num_classes=NUM_CLASSES)
    global_model, _ = prune_cnn(global_model, dropout_rate=1 - capacity, scaling=False)
elif MODEL_TYPE == "resnet":
    global_model = resnet18(weights=None, num_classes=NUM_CLASSES)
    replace_bn_with_ln(global_model)
    global_model, _ = prune_resnet18(
        global_model, dropout_rate=1 - capacity, scaling=False
    )
else:
    raise ValueError(f"Model type {MODEL_TYPE} not supported.")
print(f"[Model Architecture]\n{global_model}")
print(f"Model Capacity: {capacity}")
print(
    f"Global Model size: {calculate_model_size(global_model, print_result=False, unit='MB'):.2f} MB"
)


# <======================================== Training, aggregation and evluation ========================================>
train_loss_results = []
test_loss_results = []
test_acc_results = []
class_wise_results = []
local_validation_results = []

client_evaluation_results = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

start_time = time()
for round in range(ROUNDS):
    print(f"Round {round + 1}")
    round_start_time = time()

    round_train_loss_results = {"Round": round + 1}
    round_test_loss_results = {"Round": round + 1}
    round_test_acc_results = {"Round": round + 1}
    round_class_wise_results = {"Round": round + 1}

    # Radomly select a ratio of clients to participate in this round
    selected_client_ids = np.random.choice(
        range(NUM_CLIENTS), int(NUM_CLIENTS * SELECT_RATIO), replace=False
    )
    print(f"Selected {len(selected_client_ids)} client IDs: {selected_client_ids}")

    # Load global model's parameters
    all_client_models = {}
    for client_id in selected_client_ids:
        all_client_models[client_id] = copy.deepcopy(global_model)

    # <---------------------------------------- Local Training ---------------------------------------->
    for client_id in selected_client_ids:
        optimizer = optim.Adam(
            all_client_models[client_id].parameters(), lr=LR, weight_decay=5e-4
        )
        local_train_loss = train(
            model=all_client_models[client_id],
            optimizer=optimizer,
            criterion=criterion,
            dataloader=train_loaders[client_id],
            device=device,
            epochs=EPOCHS,
            verbose=False,
        )
        local_evaluation_result_on_train_data = evaluate_acc(
            model=all_client_models[client_id],
            dataloader=train_loaders[client_id],
            device=device,
            class_wise=True,
        )
        local_evaluation_result = evaluate_acc(
            model=all_client_models[client_id],
            dataloader=val_loaders[client_id],
            device=device,
            class_wise=True,
        )
        client_global_evaluation_result = evaluate_acc(
            model=all_client_models[client_id],
            dataloader=global_test_loader,
            device=device,
            class_wise=True,
        )

        local_test_loss = local_evaluation_result["loss"]
        local_test_acc = local_evaluation_result["accuracy"]
        local_class_acc = local_evaluation_result["class_wise_accuracy"]
        model_size = calculate_model_size(
            all_client_models[client_id], print_result=False, unit="MB"
        )

        client_evaluation_results[client_id] = {
            "model_size": model_size,
            "train_loss": local_train_loss,
            "train_acc": local_evaluation_result_on_train_data["accuracy"],
            "local_val_loss": local_test_loss,
            "local_val_acc": local_test_acc,
            "global_val_loss": client_global_evaluation_result["loss"],
            "global_val_acc": client_global_evaluation_result["accuracy"],
        }

        round_train_loss_results[f"Client {client_id}"] = local_train_loss
        round_test_loss_results[f"Client {client_id}"] = local_test_loss
        round_test_acc_results[f"Client {client_id}"] = local_test_acc
        round_class_wise_results[f"Client {client_id}"] = local_class_acc

    # <---------------------------------------- Aggregation ---------------------------------------->
    client_weights = [subset_sizes[i] for i in selected_client_ids]
    if MODEL_TYPE == "cnn":
        aggregated_weight = vanilla_federated_averaging(
            models=[model for model in all_client_models.values()],
            client_weights=client_weights,
        )
        global_model.load_state_dict(aggregated_weight)
    elif MODEL_TYPE == "resnet":
        aggregated_weight = vanilla_federated_averaging(
            models=[model for model in all_client_models.values()],
            client_weights=client_weights,
        )
        global_model.load_state_dict(aggregated_weight)
    else:
        raise ValueError(f"Model type {MODEL_TYPE} not supported.")

# <---------------------------------------- Global Model Evaluation ---------------------------------------->
    global_evaluation_result = evaluate_acc(
        model=global_model,
        dataloader=global_test_loader,
        device=device,
        class_wise=True,
    )
    client_evaluation_results["aggregated"] = {
        "global_val_loss": global_evaluation_result["loss"],
        "global_val_acc": global_evaluation_result["accuracy"],
    }
    # Evaluate the global model on each selected local validation set
    for client_id in selected_client_ids:
        aggregated_evaluation_result_on_local_validation_data = evaluate_acc(
            model=global_model,
            dataloader=val_loaders[client_id],
            device=device,
            class_wise=True,
        )
        client_evaluation_results["aggregated"][client_id] = {
            "local_val_loss": aggregated_evaluation_result_on_local_validation_data[
                "loss"
            ],
            "local_val_acc": aggregated_evaluation_result_on_local_validation_data[
                "accuracy"
            ],
        }

    # <---------------------------------------- Print Results ---------------------------------------->
    for client_id in selected_client_ids:
        print(
            f"Client {client_id}\t"
            f"Model Size: {client_evaluation_results[client_id]["model_size"]:.2f} MB\t"
            f"Train Loss: {client_evaluation_results[client_id]["train_loss"]:.4f}\t"
            f"Train Acc: {client_evaluation_results[client_id]["train_acc"]:.4f}\t"
            f"Local Validation  Loss: {client_evaluation_results[client_id]["local_val_loss"]:.4f}\t"
            f"Local Validation Acc: {client_evaluation_results[client_id]["local_val_acc"]:.4f}\t"
            f"Global Validation Loss: {client_evaluation_results[client_id]["global_val_loss"]:.4f}\t"
            f"Global Validation Acc: {client_evaluation_results[client_id]["global_val_acc"]:.4f}"
        )

    global_test_loss = global_evaluation_result["loss"]
    global_test_acc = global_evaluation_result["accuracy"]
    global_class_acc = global_evaluation_result["class_wise_accuracy"]
    print(
        "Aggregated\n"
        f"Global Test Loss: {global_test_loss:.4f}\t"
        f"Global Test Acc: {global_test_acc:.4f}"
    )
    for client_id in selected_client_ids:
        local_val_loss = client_evaluation_results["aggregated"][client_id]["local_val_loss"]
        loss_gap = local_val_loss - client_evaluation_results[client_id]["local_val_loss"]
        local_val_acc = client_evaluation_results["aggregated"][client_id]["local_val_acc"]
        acc_gap = local_val_acc - client_evaluation_results[client_id]["local_val_acc"]
        print(
            f"Client {client_id}\t"
            f"Local Validation Loss: {local_val_loss:.4f}({loss_gap:.4f})\t"
            f"Local Validation Acc: {local_val_acc:.4f}({acc_gap:.4f})"
        )

    round_test_loss_results["Aggregated"] = global_test_loss
    round_test_acc_results["Aggregated"] = global_test_acc
    round_class_wise_results["Aggregated"] = global_class_acc

    train_loss_results.append(round_train_loss_results)
    test_loss_results.append(round_test_loss_results)
    test_acc_results.append(round_test_acc_results)
    class_wise_results.append(round_class_wise_results)

    round_end_time = time()
    round_use_time = round_end_time - round_start_time
    print(
        f"Round {round + 1} use time: {round_use_time/60:.2f} min, ETA: {(ROUNDS - round - 1) * round_use_time / 3600:.2f} hours"
    )
    print("=" * 80)

end_time = time()
print(f"Total use time: {(end_time - start_time) / 3600:.2f} hours")


# <==================== Save results to files ====================>
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
# class_wise_results = format_results(class_wise_results)
local_validation_results = format_results(local_validation_results)

if SAVE_DIR is not None:
    with open(
        os.path.join(
            SAVE_DIR, f"{METHOD}_{MODEL_TYPE}_{DATASET}_{args.alpha}_train_loss.csv"
        ),
        "w",
    ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=train_loss_results[0].keys())
        writer.writeheader()
        writer.writerows(train_loss_results)

    with open(
        os.path.join(
            SAVE_DIR, f"{METHOD}_{MODEL_TYPE}_{DATASET}_{args.alpha}_test_loss.csv"
        ),
        "w",
    ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=test_loss_results[0].keys())
        writer.writeheader()
        writer.writerows(test_loss_results)

    with open(
        os.path.join(
            SAVE_DIR, f"{METHOD}_{MODEL_TYPE}_{DATASET}_{args.alpha}_test_acc.csv"
        ),
        "w",
    ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=test_acc_results[0].keys())
        writer.writeheader()
        writer.writerows(test_acc_results)

    with open(
        os.path.join(
            SAVE_DIR, f"{METHOD}_{MODEL_TYPE}_{DATASET}_{args.alpha}_class_acc.pkl"
        ),
        "wb",
    ) as f:
        pickle.dump(class_wise_results, f)

    with open(
        os.path.join(
            SAVE_DIR,
            f"{METHOD}_{MODEL_TYPE}_{DATASET}_{args.alpha}_local_validation.csv",
        ),
        "w",
    ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=local_validation_results[0].keys())
        writer.writeheader()
        writer.writerows(local_validation_results)

    print("Results saved.")
else:
    print("Results not saved.")
