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
from modules.constants import (
    NORMALIZATION_STATS,
    OPTIONAL_CLIENT_CAPACITY_DISTRIBUTIONS,
)
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
if args.dataset == "cifar10":
    NUM_CLASSES = 10
elif args.dataset == "cifar100":
    NUM_CLASSES = 100
elif args.dataset == "tiny-imagenet":
    NUM_CLASSES = 200
else:
    raise ValueError(f"Dataset {args.dataset} not supported.")


# <======================================== Set random seed for reproducibility ========================================>
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# <======================================== Data preparation ========================================>
if args.data_augmentation:
    if "cifar" in args.dataset:
        input_image_size = 32
    elif args.dataset == "tiny-imagenet":
        input_image_size = 64
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(input_image_size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=NORMALIZATION_STATS[args.dataset]["mean"],
                std=NORMALIZATION_STATS[args.dataset]["std"],
            ),
        ]
    )
else:
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=NORMALIZATION_STATS[args.dataset]["mean"],
                std=NORMALIZATION_STATS[args.dataset]["std"],
            ),
        ]
    )

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            NORMALIZATION_STATS[args.dataset]["mean"],
            NORMALIZATION_STATS[args.dataset]["std"],
        ),
    ]
)

if args.dataset == "cifar10":
    DatasetClass = datasets.CIFAR10
elif args.dataset == "cifar100":
    DatasetClass = datasets.CIFAR100
elif args.dataset == "tiny-imagenet":
    DatasetClass = TinyImageNet
else:
    raise ValueError(f"Dataset {args.dataset} not supported.")

global_train_dataset = DatasetClass(
    root="./data", train=True, transform=train_transform
)
global_test_dataset = DatasetClass(root="./data", train=False, transform=test_transform)
global_test_loader = DataLoader(
    global_test_dataset, batch_size=args.batch_size, shuffle=False
)

if args.distribution == "iid":
    num_subsets = args.num_clients
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
    num_clients = args.num_clients
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
        plt.savefig(f"num_samples_distribution_{args.dataset}_alpha{alpha}.png")

        # Plot the class distribution of each client
        num_classes = len(class_distributions[0])
        ind = np.arange(num_classes)
        width = 0.5
        colors = plt.get_cmap("tab20", len(client_ids))
        bottom = np.zeros(num_classes)
        plt.figure(figsize=(16, 9))
        for i, client_id in enumerate(client_ids):
            class_counts = list(class_distributions[i].values())
            plt.bar(
                ind,
                class_counts,
                width,
                label=f"Client {client_id}",
                color=colors(i),
                bottom=bottom,
            )
            bottom += class_counts
        plt.xlabel("Class")
        plt.ylabel("Number of Samples")
        if args.dataset == "tiny-imagenet":
            xticks = np.arange(0, num_classes, step=2)
        else:
            xticks = np.arange(0, num_classes)
        xtick_labels = [str(i) for i in xticks]
        plt.xticks(xticks, xtick_labels, rotation=90)
        plt.legend(bbox_to_anchor=(0.5, -0.1), loc="upper center", ncol=10)

        plt.gca().margins(
            x=0
        )  # Adjust the margins to reduce space between bars and y-axis
        plt.savefig(
            f"class_distribution_{args.dataset}_alpha{alpha}.png", bbox_inches="tight"
        )
        exit()
else:
    raise ValueError(f"Data distribution {args.distribution} not supported.")

# Split every client's data into train and validation sets
train_ratio = args.local_train_ratio
train_loaders = []
if args.local_train_ratio < 1.0:
    val_loaders = []
for indices_a_subset in subset_indices_list:
    split_point = int(train_ratio * len(indices_a_subset))
    np.random.shuffle(indices_a_subset)
    train_indices = indices_a_subset[:split_point]
    train_loaders.append(
        DataLoader(
            Subset(global_train_dataset, train_indices),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
    )
    if args.local_train_ratio < 1.0:
        val_indices = indices_a_subset[split_point:]
        val_loaders.append(
            DataLoader(
                Subset(global_train_dataset, val_indices),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
        )


# <======================================== Model preparation ========================================>
if args.model == "cnn":
    global_model = CNN(num_classes=NUM_CLASSES)
elif args.model == "resnet":
    if args.norm_type == "sbn":
        global_model = custom_resnet18(
            num_classes=NUM_CLASSES, weights=None, norm_type=args.norm_type
        )
    elif args.norm_type == "ln":
        input_shape = [input_image_size, input_image_size]
        global_model = custom_resnet18(
            num_classes=NUM_CLASSES,
            weights=None,
            norm_type=args.norm_type,
            input_shape=input_shape,
        )
    else:
        raise ValueError(f"Normalization type {args.norm_type} not supported.")
else:
    raise ValueError(f"Model type {args.model} not supported.")
print(f"[Model Architecture]\n{global_model}")
print(
    f"Global Model size: {calculate_model_size(global_model, print_result=False, unit='MB'):.2f} MB"
)


# <======================================== Server preparation ========================================>
client_capacity_distribution = OPTIONAL_CLIENT_CAPACITY_DISTRIBUTIONS[
    args.client_capacity_distribution
]
optional_client_capacities = [t[0] for t in client_capacity_distribution]
weights = [t[1] for t in client_capacity_distribution]
if args.method in ["fedavg"]:
    capacity_counts = [args.num_clients]
    min_capacity = min(optional_client_capacities)
    client_capacities = [min_capacity] * args.num_clients
else:
    capacity_counts = [int(args.num_clients * w) for w in weights]
    while sum(capacity_counts) < args.num_clients:
        for i in range(len(capacity_counts)):
            capacity_counts[i] += 1
            if sum(capacity_counts) == args.num_clients:
                break
    client_capacities = []
    for capacity, count in zip(optional_client_capacities, capacity_counts):
        client_capacities.extend([capacity] * count)
    random.shuffle(client_capacities)
print(f"Optional client capacities: {optional_client_capacities}")
print(f"Weights: {weights}")
print(f"Capacity counts: {capacity_counts}")
print(f"Specific client capacities: {client_capacities}")

if abs(args.gamma - 0.9) < 1e-6:
    decay_steps = [i for i in range(10, 201, 10)]
elif abs(args.gamma - 0.8) < 1e-6:
    decay_steps = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
elif abs(args.gamma - 0.5) < 1e-6:
    decay_steps = [50, 100]
else:
    decay_steps = []
print(f"Decay steps: {decay_steps}")

if args.method == "fedavg":
    server = ServerHomo(
        global_model=global_model,
        dataset=args.dataset,
        num_clients=args.num_clients,
        client_capacities=client_capacities,
        model_out_dim=NUM_CLASSES,
        model_type=args.model,
        select_ratio=args.client_select_ratio,
        norm_type=args.norm_type,
        eta_g=args.eta_g,
        dynamic_eta_g=args.dynamic_eta_g,
        global_lr_decay=args.global_lr_decay,
        gamma=args.gamma,
        decay_steps=decay_steps,
    )
elif args.method == "heterofl":
    server = ServerStatic(
        global_model=global_model,
        dataset=args.dataset,
        num_clients=args.num_clients,
        client_capacities=client_capacities,
        model_out_dim=NUM_CLASSES,
        model_type=args.model,
        select_ratio=args.client_select_ratio,
        scaling=True,
        norm_type=args.norm_type,
        eta_g=args.eta_g,
        dynamic_eta_g=args.dynamic_eta_g,
        param_delta_norm=args.param_delta_norm,
        global_lr_decay=args.global_lr_decay,
        gamma=args.gamma,
        decay_steps=decay_steps,
    )
elif args.method == "fedrolex":
    server = ServerFedRolex(
        global_model=global_model,
        dataset=args.dataset,
        num_clients=args.num_clients,
        client_capacities=client_capacities,
        model_out_dim=NUM_CLASSES,
        model_type=args.model,
        select_ratio=args.client_select_ratio,
        scaling=True,
        norm_type=args.norm_type,
        param_delta_norm=args.param_delta_norm,
        global_lr_decay=args.global_lr_decay,
        gamma=args.gamma,
        decay_steps=decay_steps,
        rolling_step=-1,
    )
elif args.method == "fedrd":
    server = ServerRD(
        global_model=global_model,
        dataset=args.dataset,
        num_clients=args.num_clients,
        client_capacities=client_capacities,
        model_out_dim=NUM_CLASSES,
        model_type=args.model,
        select_ratio=args.client_select_ratio,
        scaling=True,
        norm_type=args.norm_type,
        eta_g=args.eta_g,
        dynamic_eta_g=args.dynamic_eta_g,
        param_delta_norm=args.param_delta_norm,
        global_lr_decay=args.global_lr_decay,
        gamma=args.gamma,
        decay_steps=decay_steps,
    )
elif args.method == "fedrame":
    server = ServerFedRAME(
        global_model=global_model,
        dataset=args.dataset,
        num_clients=args.num_clients,
        client_capacities=client_capacities,
        model_out_dim=NUM_CLASSES,
        model_type=args.model,
        select_ratio=args.client_select_ratio,
        scaling=True,
        norm_type=args.norm_type,
        eta_g=args.eta_g,
        dynamic_eta_g=args.dynamic_eta_g,
        param_delta_norm=args.param_delta_norm,
        global_lr_decay=args.global_lr_decay,
        gamma=args.gamma,
        decay_steps=decay_steps,
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
for round in range(args.rounds):
    round_start_time = time()
    print(f"Round {round + 1}")

    all_client_models = {}
    model_pruned_indices_dicts = {}

    # <---------------------------------------- Submodel Distribution ---------------------------------------->
    if args.method in ["fedrolex", "heterofl", "fedrd", "fedrame"]:
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
    elif args.method in ["fedavg"]:
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
            all_client_models[client_id].parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        # Train the client model
        local_train_loss = train(
            model=all_client_models[client_id],
            optimizer=optimizer,
            criterion=criterion,
            dataloader=train_loaders[client_id],
            device=device,
            epochs=args.epochs,
            verbose=False,
        )

        # Evaluate the client model
        local_evaluation_result_on_train_data = evaluate_acc(
            model=all_client_models[client_id],
            dataloader=train_loaders[client_id],
            device=device,
            class_wise=True,
        )
        if args.local_train_ratio < 1.0:
            local_evaluation_result = evaluate_acc(
                model=all_client_models[client_id],
                dataloader=val_loaders[client_id],
                device=device,
                class_wise=True,
            )
        else:
            local_evaluation_result = {}
        client_global_evaluation_result = evaluate_acc(
            model=all_client_models[client_id],
            dataloader=global_test_loader,
            device=device,
            class_wise=True,
        )

        # Store the evaluation results of the client for later printing
        model_size = calculate_model_size(
            all_client_models[client_id], print_result=False, unit="MB"
        )
        client_evaluation_results[client_id] = {
            "model_size": model_size,
            "train_loss": local_train_loss,
            "train_acc": local_evaluation_result_on_train_data["accuracy"],
            "local_val_loss": local_evaluation_result.get("loss", 0.0),
            "local_val_acc": local_evaluation_result.get("accuracy", 0.0),
            "global_val_loss": client_global_evaluation_result["loss"],
            "global_val_acc": client_global_evaluation_result["accuracy"],
        }

    # <---------------------------------------- Aggregation ---------------------------------------->
    if args.method in ["fedavg"]:
        server.step(
            local_state_dicts=[
                model.state_dict() for model in all_client_models.values()
            ],
            selected_client_ids=selected_client_ids,
        )
    elif args.method in ["fedrolex", "heterofl", "fedrd", "fedrame"]:
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

    if args.local_train_ratio < 1.0:
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
            f"Capacity: {client_capacities[client_id]}\t"
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
    if args.local_train_ratio < 1.0:
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
            acc_gap = (
                local_val_acc - client_evaluation_results[client_id]["local_val_acc"]
            )
            print(
                f"Client {client_id}\t"
                f"Local Validation Loss: {local_val_loss:.4f}({loss_gap:.4f})\t"
                f"Local Validation Acc: {local_val_acc:.4f}({acc_gap:.4f})"
            )

    # Estimate ETA
    round_end_time = time()
    round_use_time = round_end_time - round_start_time
    print(
        f"Round {round + 1} use time: {round_use_time/60:.2f} min, ETA: {(args.rounds - round - 1) * round_use_time / 3600:.2f} hours"
    )
    print("=" * 80)

end_time = time()
print(f"Total use time: {(end_time - start_time) / 3600:.2f} hours")
