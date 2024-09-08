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
from modules.aggregation.aggregate_models import aggregate_resnet18_vanilla
from modules.args_parser import get_args
from modules.constants import NORMALIZATION_STATS
from modules.data import create_non_iid_data
from modules.debugging import create_empty_pruned_indices_dict
from modules.evaluation import test
from modules.models import CNN
from modules.pruning import prune_cnn, prune_resnet18
from modules.training import train
from modules.utils import calculate_model_size, replace_bn_with_ln

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
ROUNDS = args.round
EPOCHS = args.epochs
LR = args.lr
BATCH_SIZE = args.batch_size
NUM_CLIENTS = args.num_clients

seed = 18
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            NORMALIZATION_STATS[DATASET]["mean"], NORMALIZATION_STATS[DATASET]["std"]
        ),
    ]
)

if DATASET == "cifar10":
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=False, transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=False, transform=transform
    )
elif DATASET == "cifar100":
    train_dataset = datasets.CIFAR100(
        root="./data", train=True, download=False, transform=transform
    )

    test_dataset = datasets.CIFAR100(
        root="./data", train=False, download=False, transform=transform
    )
else:
    raise ValueError(f"Dataset {DATASET} not supported.")

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

if args.distribution == "iid":
    num_train = len(train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)

    num_subsets = NUM_CLIENTS
    subset_size = num_train // num_subsets
    subsets_indices = [
        indices[i : i + subset_size] for i in range(0, num_train, subset_size)
    ]
    subset_sizes = [len(subset) for subset in subsets_indices]
elif args.distribution == "non-iid":
    num_clients = NUM_CLIENTS
    alpha = args.alpha
    client_data_indices, client_stats = create_non_iid_data(
        train_dataset, num_clients, alpha
    )
    subsets_indices = list(client_data_indices.values())
    subset_sizes = [len(subset) for subset in subsets_indices]

    client_ids = list(client_data_indices.keys())
    total_samples = [client_stats[i]["total_samples"] for i in client_ids]
    class_distributions = [client_stats[i]["class_distribution"] for i in client_ids]

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
        # plt.bar(ind + i * width, class_counts, width, label=f"Client {client_id}")
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
    # plt.xticks(ind + width * (num_clients // 2), [str(i) for i in range(NUM_CLASSES)])
    plt.xticks(ind, [str(i) for i in range(NUM_CLASSES)])
    # plt.legend()
    # plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.legend(bbox_to_anchor=(0.5, -0.1), loc="upper center", ncol=10)
    plt.savefig(f"class_distribution_{DATASET}_{alpha}.png", bbox_inches="tight")
else:
    raise ValueError(f"Data distribution {args.distribution} not supported.")

dataloaders = [
    DataLoader(
        Subset(train_dataset, subset_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    for subset_indices in subsets_indices
]

if MODEL_TYPE == "cnn":
    original_cnn = CNN(num_classes=NUM_CLASSES)
elif MODEL_TYPE == "resnet":
    original_cnn = resnet18(weights=None, num_classes=NUM_CLASSES)
    replace_bn_with_ln(original_cnn)
else:
    raise ValueError(f"Model type {MODEL_TYPE} not supported.")
print(f"[Model Architecture]\n{original_cnn}")

p = 0.8
if MODEL_TYPE == "cnn":
    # global_cnn, _ = prune_cnn(original_cnn, p, position=0)
    global_cnn, _ = prune_cnn(original_cnn, p)
elif MODEL_TYPE == "resnet":
    global_cnn, _ = prune_resnet18(original_cnn, p)
else:
    raise ValueError(f"Model type {MODEL_TYPE} not supported.")

num_models = NUM_CLIENTS
all_client_models = []
for i in range(num_models):
    if MODEL_TYPE == "cnn":
        client_model, _ = prune_cnn(original_cnn, p)
    elif MODEL_TYPE == "resnet":
        client_model, _ = prune_resnet18(original_cnn, p)
    else:
        raise ValueError(f"Model type {MODEL_TYPE} not supported.")
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
            local_model, criterion, test_loader, device, num_classes=NUM_CLASSES
        )
        model_size = calculate_model_size(local_model, print_result=False, unit="MB")
        print(
            f"Subset {i + 1}\tModel Size: {model_size:.2f} MB\tTrain Loss: {local_train_loss:.4f}\tTest Loss: {local_test_loss:.4f}\tTest Acc: {local_test_acc:.4f}"
        )
        round_train_loss_results[f"Subset {i + 1}"] = local_train_loss
        round_test_loss_results[f"Subset {i + 1}"] = local_test_loss
        round_test_acc_results[f"Subset {i + 1}"] = local_test_acc
        round_class_acc_results[f"Subset {i + 1}"] = local_class_acc

    # Aggregation
    if MODEL_TYPE == "cnn":
        aggregated_weight = vanilla_federated_averaging(
            models=all_client_models, sample_numbers=subset_sizes
        )
        global_cnn.load_state_dict(aggregated_weight)
    elif MODEL_TYPE == "resnet":
        # aggregate_resnet18_vanilla(
        #     global_model=global_cnn,
        #     local_models=all_client_models,
        #     client_weights=subset_sizes,
        #     pruned_indices_dicts=[
        #         create_empty_pruned_indices_dict() for _ in range(num_models)
        #     ],
        # )
        aggregated_weight = vanilla_federated_averaging(
            models=all_client_models, sample_numbers=subset_sizes
        )
        global_cnn.load_state_dict(aggregated_weight)
    else:
        raise ValueError(f"Model type {MODEL_TYPE} not supported.")

    global_test_loss, global_test_acc, global_class_acc = test(
        global_cnn, criterion, test_loader, device, num_classes=NUM_CLASSES
    )
    print(
        f"Aggregated Test Loss: {global_test_loss:.4f}\tAggregated Test Acc: {global_test_acc:.4f}"
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

if SAVE_DIR is not None:
    with open(
        os.path.join(SAVE_DIR, f"vanilla_{MODEL_TYPE}_{DATASET}_train_loss.csv"), "w"
    ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=train_loss_results[0].keys())
        writer.writeheader()
        writer.writerows(train_loss_results)

    with open(
        os.path.join(SAVE_DIR, f"vanilla_{MODEL_TYPE}_{DATASET}_test_loss.csv"), "w"
    ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=test_loss_results[0].keys())
        writer.writeheader()
        writer.writerows(test_loss_results)

    with open(
        os.path.join(SAVE_DIR, f"vanilla_{MODEL_TYPE}_{DATASET}_test_acc.csv"), "w"
    ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=test_acc_results[0].keys())
        writer.writeheader()
        writer.writerows(test_acc_results)

    with open(
        os.path.join(SAVE_DIR, f"vanilla_{MODEL_TYPE}_{DATASET}_class_acc.pkl"), "wb"
    ) as f:
        pickle.dump(class_acc_results, f)

    print("Results saved.")
else:
    print("Results not saved.")
