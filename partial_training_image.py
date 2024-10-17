import random
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from modules.args_parser import get_args
from modules.constants import NORMALIZATION_STATS, OPTIONAL_CLIENT_CAPACITY_DISTRIBUTIONS
from modules.data import TinyImageNet, create_non_iid_data
from modules.evaluation import evaluate_acc
from modules.models import CNN, custom_resnet18
from modules.servers import (
    ServerFedRAME,
    ServerFedRolex,
    ServerHomo,
    ServerRD,
    ServerStatic,
)
from modules.training import train
from modules.utils import (
    calculate_model_size,
)

# <======================================== Parse arguments ========================================>
args = get_args()
METHOD = args.method
MODEL_TYPE = args.model
DATASET = args.dataset
if DATASET == "cifar10":
    NUM_CLASSES = 10
elif DATASET == "cifar100":
    NUM_CLASSES = 100
elif DATASET == "tiny-imagenet":
    NUM_CLASSES = 200
else:
    raise ValueError(f"Dataset {DATASET} not supported.")
NUM_CLIENTS = args.num_clients
SELECT_RATIO = args.select_ratio
LOCAL_VALIDATION_FREQUENCY = int(1 / args.select_ratio)
LOCAL_TRAIN_RATIO = args.local_train_ratio
ROUNDS = args.rounds
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr

# Set random seed for reproducibility
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# <======================================== Data preparation ========================================>
if "cifar" in DATASET:
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(
                NORMALIZATION_STATS[DATASET]["mean"], NORMALIZATION_STATS[DATASET]["std"]
            ),
        ]
    )
elif DATASET == "tiny-imagenet":
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=NORMALIZATION_STATS[DATASET]["mean"],
                std=NORMALIZATION_STATS[DATASET]["std"],
            ),
        ]
    )
else:
    raise ValueError(f"Dataset {DATASET} not supported.")

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
elif DATASET == "tiny-imagenet":
    global_train_dataset = TinyImageNet(
        root_dir="./data/tiny-imagenet-200", split="train", transform=transform
    )

    global_test_dataset = TinyImageNet(
        root_dir="./data/tiny-imagenet-200", split="val", transform=transform
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
elif isinstance(args.distribution, float):
    num_clients = NUM_CLIENTS
    alpha = args.distribution
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
    train_loaders.append(
        DataLoader(
            Subset(global_train_dataset, train_indices),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=args.num_workers,
        )
    )
    val_loaders.append(
        DataLoader(
            Subset(global_train_dataset, val_indices),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=args.num_workers,
        )
    )


# <======================================== Model preparation ========================================>
if MODEL_TYPE == "cnn":
    global_model = CNN(num_classes=NUM_CLASSES)
elif MODEL_TYPE == "resnet":
    if args.norm_type == "sbn":
        global_model = custom_resnet18(
            num_classes=NUM_CLASSES, weights=None, norm_type=args.norm_type
        )
    elif args.norm_type == "ln":
        if "cifar" in DATASET:
            input_shape = [32, 32]
        elif DATASET == "tiny-imagenet":
            input_shape = [64, 64]
        else:
            raise ValueError(f"Dataset {DATASET} not supported.")
        global_model = custom_resnet18(
            num_classes=NUM_CLASSES,
            weights=None,
            norm_type=args.norm_type,
            input_shape=input_shape,
        )
    else:
        raise ValueError(f"Normalization type {args.norm_type} not supported.")
else:
    raise ValueError(f"Model type {MODEL_TYPE} not supported.")
print(f"[Model Architecture]\n{global_model}")
print(
    f"Global Model size: {calculate_model_size(global_model, print_result=False, unit='MB'):.2f} MB"
)


# <======================================== Server preparation ========================================>
client_capacity_distribution = OPTIONAL_CLIENT_CAPACITY_DISTRIBUTIONS[args.client_capacity_distribution]
client_capacities = [t[0] for t in client_capacity_distribution]
optional_dropout_rates = [1 - capacity for capacity in client_capacities]
weights = [t[1] for t in client_capacity_distribution]
if METHOD in ["fedavg"]:
    max_dropout_rate = max(optional_dropout_rates)
    client_dropout_rates = [max_dropout_rate] * NUM_CLIENTS
else:
    client_dropout_rates = [
        random.choices(optional_dropout_rates, weights=weights)[0]
        for _ in range(NUM_CLIENTS)
    ]
dropout_rate_count = {}
for rate in client_dropout_rates:
    if rate in dropout_rate_count:
        dropout_rate_count[rate] += 1
    else:
        dropout_rate_count[rate] = 1
print(f"Possible client dropout rates: {optional_dropout_rates}")
print(f"Weights: {weights}")
print(f"Client dropout rates: {client_dropout_rates}")
print(f"Client dropout rate count: {dropout_rate_count}")

if METHOD == "fedavg":
    server = ServerHomo(
        global_model=global_model,
        dataset=DATASET,
        num_clients=NUM_CLIENTS,
        client_capacity=min([1.0 - p for p in client_dropout_rates]),
        model_out_dim=NUM_CLASSES,
        model_type=MODEL_TYPE,
        select_ratio=SELECT_RATIO,
        scaling=True,
        norm_type=args.norm_type,
    )
elif METHOD == "heterofl":
    server = ServerStatic(
        global_model=global_model,
        dataset=DATASET,
        num_clients=NUM_CLIENTS,
        client_capacities=[1.0 - p for p in client_dropout_rates],
        model_out_dim=NUM_CLASSES,
        model_type=MODEL_TYPE,
        select_ratio=SELECT_RATIO,
        scaling=True,
        norm_type=args.norm_type,
    )
elif METHOD == "fedrolex":
    server = ServerFedRolex(
        global_model=global_model,
        dataset=DATASET,
        num_clients=NUM_CLIENTS,
        client_capacities=[1.0 - p for p in client_dropout_rates],
        model_out_dim=NUM_CLASSES,
        model_type=MODEL_TYPE,
        select_ratio=SELECT_RATIO,
        scaling=True,
        rolling_step=-1,
        norm_type=args.norm_type,
    )
elif METHOD == "fedrd":
    server = ServerRD(
        global_model=global_model,
        dataset=DATASET,
        num_clients=NUM_CLIENTS,
        client_capacities=[1.0 - p for p in client_dropout_rates],
        model_out_dim=NUM_CLASSES,
        model_type=MODEL_TYPE,
        select_ratio=SELECT_RATIO,
        scaling=True,
        norm_type=args.norm_type,
    )
elif METHOD == "fedrame":
    server = ServerFedRAME(
        global_model=global_model,
        dataset=DATASET,
        num_clients=NUM_CLIENTS,
        client_capacities=[1.0 - p for p in client_dropout_rates],
        model_out_dim=NUM_CLASSES,
        model_type=MODEL_TYPE,
        select_ratio=SELECT_RATIO,
        scaling=True,
        eta_g=args.eta_g,
        dynamic_eta_g=args.dynamic_eta_g,
        norm_type=args.norm_type,
        shrinking=args.shrinking,
    )


# <======================================== Training, aggregation and evaluation ========================================>
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
    round_start_time = time()
    print(f"Round {round + 1}")

    all_client_models = {}
    model_pruned_indices_dicts = {}

    # <---------------------------------------- Submodel Distribution ---------------------------------------->
    if METHOD in ["fedrolex", "heterofl", "fedrd", "fedrame"]:
        distributed = server.distribute()
        selected_client_ids = distributed["client_ids"]
        selected_client_capacities = distributed["client_capacities"]
        selected_submodel_param_indices_dicts = distributed[
            "submodel_param_indices_dicts"
        ]
        selected_client_submodels = distributed["client_submodels"]
        all_client_models = {
            client_id: model
            for client_id, model in zip(selected_client_ids, selected_client_submodels)
        }
    elif METHOD in ["fedavg"]:
        distributed = server.distribute()
        selected_client_ids = distributed["client_ids"]
        selected_client_submodels = distributed["client_submodels"]
        all_client_models = {
            client_id: model
            for client_id, model in zip(selected_client_ids, selected_client_submodels)
        }
    else:
        raise NotImplementedError

    print(f"Selected {len(selected_client_ids)} client IDs: {selected_client_ids}")

    # <---------------------------------------- Local Training ---------------------------------------->
    for i, client_id in enumerate(selected_client_ids):
        optimizer = optim.Adam(
            all_client_models[client_id].parameters(), lr=LR, weight_decay=5e-4
        )

        # Train the client model
        local_train_loss = train(
            model=all_client_models[client_id],
            optimizer=optimizer,
            criterion=criterion,
            dataloader=train_loaders[client_id],
            device=device,
            epochs=EPOCHS,
            verbose=False,
        )

        # Evaluate the client model
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

        # Store the evaluation results of the client for later printing
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

    # <---------------------------------------- Aggregation ---------------------------------------->
    if METHOD in ["fedavg"]:
        server.step(
            local_state_dicts=[
                model.state_dict() for model in all_client_models.values()
            ],
            selected_client_ids=selected_client_ids,
        )
    elif METHOD in ["fedrolex", "heterofl", "fedrd", "rdbagging", "fedrame"]:
        server.step(
            local_state_dicts=[
                model.state_dict() for model in all_client_models.values()
            ],
            selected_client_ids=selected_client_ids,
            submodel_param_indices_dicts=selected_submodel_param_indices_dicts,
        )
    else:
        raise NotImplementedError

    # <---------------------------------------- Global Model Evaluation ---------------------------------------->
    global_evaluation_result = evaluate_acc(
        model=server.global_model,
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
            model=server.global_model,
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
    # Print client evaluation results
    for client_id in selected_client_ids:
        print(
            f"Client {client_id}\t"
            f"Dropout Rate: {client_dropout_rates[client_id]}\t"
            f"Model Size: {client_evaluation_results[client_id]['model_size']:.2f} MB\t"
            f"Train Loss: {client_evaluation_results[client_id]['train_loss']:.4f}\t"
            f"Train Acc: {client_evaluation_results[client_id]['train_acc']:.4f}\t"
            f"Local Validation  Loss: {client_evaluation_results[client_id]['local_val_loss']:.4f}\t"
            f"Local Validation Acc: {client_evaluation_results[client_id]['local_val_acc']:.4f}\t"
            f"Global Validation Loss: {client_evaluation_results[client_id]['global_val_loss']:.4f}\t"
            f"Global Validation Acc: {client_evaluation_results[client_id]['global_val_acc']:.4f}"
        )

    # Print aggregated evaluation results
    print(
        "Aggregated\n"
        f"Global Test Loss: {global_evaluation_result['loss']:.4f}\t"
        f"Global Test Acc: {global_evaluation_result['accuracy']:.4f}"
    )
    for client_id in selected_client_ids:
        local_val_loss = client_evaluation_results["aggregated"][client_id][
            "local_val_loss"
        ]
        loss_gap = (
            local_val_loss - client_evaluation_results[client_id]["local_val_loss"]
        )
        local_val_acc = client_evaluation_results["aggregated"][client_id][
            "local_val_acc"
        ]
        acc_gap = local_val_acc - client_evaluation_results[client_id]["local_val_acc"]
        print(
            f"Client {client_id}\t"
            f"Local Validation Loss: {local_val_loss:.4f}({loss_gap:.4f})\t"
            f"Local Validation Acc: {local_val_acc:.4f}({acc_gap:.4f})"
        )

    if (round + 1) % LOCAL_VALIDATION_FREQUENCY == 0 and LOCAL_TRAIN_RATIO < 1.0:
        # Evaluate the global model on **all** local validation sets
        round_local_validation_results = {"Round": round + 1}
        avg_local_validation_acc = 0.0
        dropout_rate_results = {p: [] for p in optional_dropout_rates}
        for i in range(NUM_CLIENTS):
            local_validation_result = evaluate_acc(
                model=server.global_model,
                dataloader=val_loaders[i],
                device=device,
                class_wise=True,
            )
            avg_local_validation_acc += local_validation_result["accuracy"]
            # Each client has a corresponding dropout rate
            dropout_rate = client_dropout_rates[i]
            dropout_rate_results[dropout_rate].append(
                local_validation_result["accuracy"]
            )
        avg_local_validation_acc /= NUM_CLIENTS
        round_local_validation_results["Average"] = avg_local_validation_acc
        for p in optional_dropout_rates:
            dropout_rate_results[p] = np.mean(dropout_rate_results[p])
            round_local_validation_results[f"Drpout Rate {p}"] = dropout_rate_results[p]
        local_validation_results.append(round_local_validation_results)
        # Print local validation results
        for key, value in round_local_validation_results.items():
            if key != "Round":
                print(
                    f"{key}: {value:.4f} | {value - avg_local_validation_acc:.4f} (compared to average)"
                )

    # Estimate ETA
    round_end_time = time()
    round_use_time = round_end_time - round_start_time
    print(
        f"Round {round + 1} use time: {round_use_time/60:.2f} min, ETA: {(ROUNDS - round - 1) * round_use_time / 3600:.2f} hours"
    )
    print("=" * 80)

end_time = time()
print(f"Total use time: {(end_time - start_time) / 3600:.2f} hours")
