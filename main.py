import random
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from modules.args_parser import get_args
from modules.constants import OPTIONAL_CLIENT_CAPACITY_DISTRIBUTIONS
from modules.data import (
    CIFAR10,
    CIFAR100,
    FEMNIST,
    CelebA,
    TinyImageNet,
    create_non_iid_data,
)
from modules.evaluation import evaluate_acc
from modules.models import CNN, FEMNISTCNN, custom_resnet18
from modules.servers import (
    ServerFD,
    ServerFedEcover,
    ServerFedRolex,
    ServerHomo,
    ServerStatic,
)
from modules.training import train
from modules.utils import (
    calculate_model_size,
    get_user_id_by_idx,
)

# <======================================== Parse arguments ========================================>
args = get_args()
if args.dataset == "cifar10":
    NUM_CLASSES = 10
    NUM_CLIENTS = args.num_clients
elif args.dataset == "cifar100":
    NUM_CLASSES = 100
    NUM_CLIENTS = args.num_clients
elif args.dataset == "tiny-imagenet":
    NUM_CLASSES = 200
    NUM_CLIENTS = args.num_clients
elif args.dataset == "celeba":
    NUM_CLASSES = 2
    NUM_CLIENTS = 8408
elif args.dataset == "femnist":
    NUM_CLASSES = 62
    NUM_CLIENTS = 3237
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
if "cifar" in args.dataset:
    input_image_size = 32
elif args.dataset == "tiny-imagenet":
    input_image_size = 64
elif args.dataset == "celeba":
    input_image_size = 128
elif args.dataset == "femnist":
    input_image_size = 28
else:
    raise ValueError(f"Dataset {args.dataset} not supported.")

if args.dataset == "cifar10":
    DatasetClass = CIFAR10
elif args.dataset == "cifar100":
    DatasetClass = CIFAR100
elif args.dataset == "tiny-imagenet":
    DatasetClass = TinyImageNet
elif args.dataset == "celeba":
    DatasetClass = CelebA
elif args.dataset == "femnist":
    DatasetClass = FEMNIST
else:
    raise ValueError(f"Dataset {args.dataset} not supported.")

if args.data_augmentation:
    if "cifar" in args.dataset and NUM_CLIENTS != 100:
        DATA_AUGMENTATION = False
    else:
        DATA_AUGMENTATION = True
else:
    DATA_AUGMENTATION = False

if args.dataset == "femnist" and args.model == "cnn":
    global_test_dataset = DatasetClass(
        root="./data", train=False, resize=True, augmentation=False
    )
else:
    global_test_dataset = DatasetClass(root="./data", train=False, augmentation=False)

global_test_loader = DataLoader(
    global_test_dataset, batch_size=args.batch_size, shuffle=False
)

if args.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
    global_train_dataset = DatasetClass(
        root="./data", train=True, augmentation=DATA_AUGMENTATION
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
        subset_sizes = [
            len(indices_a_subset) for indices_a_subset in subset_indices_list
        ]
    elif isinstance(args.distribution, float):
        num_clients = NUM_CLIENTS
        alpha = args.distribution
        subset_indices_list, client_stats = create_non_iid_data(
            global_train_dataset, num_clients, alpha, seed=seed
        )
        subset_sizes = [
            len(indices_a_subset) for indices_a_subset in subset_indices_list
        ]

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
                f"class_distribution_{args.dataset}_alpha{alpha}.png",
                bbox_inches="tight",
            )
            exit()
    else:
        raise ValueError(f"Data distribution {args.distribution} not supported.")

    train_loaders = []
    for indices_a_subset in subset_indices_list:
        train_loaders.append(
            DataLoader(
                Subset(global_train_dataset, indices_a_subset),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
            )
        )


# <======================================== Model preparation ========================================>
if args.model == "cnn":
    global_model = CNN(dataset=args.dataset)
elif args.model == "femnistcnn":
    global_model = FEMNISTCNN(dataset=args.dataset)
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
# Client capacity distribution setting
client_capacity_distribution = OPTIONAL_CLIENT_CAPACITY_DISTRIBUTIONS[
    args.client_capacity_distribution
]
optional_client_capacities = [t[0] for t in client_capacity_distribution]
weights = [t[1] for t in client_capacity_distribution]
if args.method in ["fedavg"]:
    capacity_counts = [NUM_CLIENTS]
    min_capacity = min(optional_client_capacities)
    client_capacities = [min_capacity] * NUM_CLIENTS
else:
    capacity_counts = [int(NUM_CLIENTS * w) for w in weights]
    while sum(capacity_counts) < NUM_CLIENTS:
        for i in range(len(capacity_counts)):
            capacity_counts[i] += 1
            if sum(capacity_counts) == NUM_CLIENTS:
                break
    client_capacities = []
    for capacity, count in zip(optional_client_capacities, capacity_counts):
        client_capacities.extend([capacity] * count)
    random.shuffle(client_capacities)
print(f"Optional client capacities: {optional_client_capacities}")
print(f"Weights: {weights}")
print(f"Capacity counts: {capacity_counts}")

# Global learning rate decay setting
decay_steps = [i for i in range(args.Ti, args.Td + 1, args.Ti)]
print(f"Decay steps: {decay_steps}")

NUM_SELECTED_CLIENTS = (
    args.client_select_num
    if args.client_select_mode == "num"
    else int(NUM_CLIENTS * args.client_select_ratio)
)

if args.method == "fedavg":
    server = ServerHomo(
        global_model=global_model,
        dataset=args.dataset,
        num_clients=NUM_CLIENTS,
        client_capacities=client_capacities,
        model_out_dim=NUM_CLASSES,
        model_type=args.model,
        num_selected_clients=NUM_SELECTED_CLIENTS,
        norm_type=args.norm_type,
        eta_g=args.eta_g,
        global_lr_decay=args.global_lr_decay,
        gamma=args.gamma,
        decay_steps=decay_steps,
    )
elif args.method == "heterofl":
    server = ServerStatic(
        global_model=global_model,
        dataset=args.dataset,
        num_clients=NUM_CLIENTS,
        client_capacities=client_capacities,
        model_out_dim=NUM_CLASSES,
        model_type=args.model,
        num_selected_clients=NUM_SELECTED_CLIENTS,
        scaling=True,
        norm_type=args.norm_type,
        eta_g=args.eta_g,
        global_lr_decay=args.global_lr_decay,
        gamma=args.gamma,
        decay_steps=decay_steps,
    )
elif args.method == "fedrolex":
    server = ServerFedRolex(
        global_model=global_model,
        dataset=args.dataset,
        num_clients=NUM_CLIENTS,
        client_capacities=client_capacities,
        model_out_dim=NUM_CLASSES,
        model_type=args.model,
        num_selected_clients=NUM_SELECTED_CLIENTS,
        scaling=True,
        norm_type=args.norm_type,
        global_lr_decay=args.global_lr_decay,
        gamma=args.gamma,
        decay_steps=decay_steps,
        rolling_step=-1,
    )
elif args.method == "fd":
    server = ServerFD(
        global_model=global_model,
        dataset=args.dataset,
        num_clients=NUM_CLIENTS,
        client_capacities=client_capacities,
        model_out_dim=NUM_CLASSES,
        model_type=args.model,
        num_selected_clients=NUM_SELECTED_CLIENTS,
        scaling=True,
        norm_type=args.norm_type,
        eta_g=args.eta_g,
        global_lr_decay=args.global_lr_decay,
        gamma=args.gamma,
        decay_steps=decay_steps,
    )
elif args.method == "fedecover":
    server = ServerFedEcover(
        global_model=global_model,
        dataset=args.dataset,
        num_clients=NUM_CLIENTS,
        client_capacities=client_capacities,
        model_out_dim=NUM_CLASSES,
        model_type=args.model,
        num_selected_clients=NUM_SELECTED_CLIENTS,
        scaling=True,
        norm_type=args.norm_type,
        eta_g=args.eta_g,
        global_lr_decay=args.global_lr_decay,
        gamma=args.gamma,
        decay_steps=decay_steps,
    )


# <======================================== Print arguments ========================================>
print("\n=== Basic Information ===\n")
print(f"Random seed: {args.seed}")
print(f"Method: {args.method}")
print(f"Model: {args.model}")
print(f"Dataset: {args.dataset}")
if args.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
    print(f"Data distribution: {args.distribution}")
else:
    print("Naturally non-IID")
print(f"Number of clients: {NUM_CLIENTS}")
print(f"Client selection mode: {args.client_select_mode}")
print(f"Client selection ratio per round: {args.client_select_ratio}")
print(f"Number of selected clients: {NUM_SELECTED_CLIENTS}")
print(f"Number of rounds: {args.rounds}")
print(f"Client capacity distribution type: {args.client_capacity_distribution}")

print("\n=== Local Training Information ===\n")
print(f"Number of local epochs: {args.epochs}")
print(f"Batch size of local training: {args.batch_size}")
print(f"Learning rate of local training: {args.lr}")
print(f"Weight decay of local training: {args.weight_decay}")
print(f"Whether use data augmentation: {DATA_AUGMENTATION}")
print(f"Number of dataloader workers: {args.num_workers}")

print("\n=== Aggregation Information ===\n")
if args.model == "resnet":
    print(f"Model normalization layer type: {args.norm_type}")
print(f"Global step-size: {args.eta_g}")
print(f"Whether use global step-size decay: {args.global_lr_decay}")
print(f"Gamma: {args.gamma}")

print("\n=== Evaluation Information ===\n")


# <======================================== Training, aggregation and evaluation ========================================>
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

client_evaluation_results = {}

start_time = time()
for round in range(args.rounds):
    round_start_time = time()
    print(f"Round {round + 1}")

    all_client_models = {}
    model_pruned_indices_dicts = {}

    # <---------------------------------------- Submodel Distribution ---------------------------------------->
    if args.method in ["fedrolex", "heterofl", "fd", "fedecover"]:
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
    distribution_time = time()
    print(f"Submodel distribution time: {distribution_time - round_start_time:.2f} sec")

    # <---------------------------------------- Local Training ---------------------------------------->
    for i, client_id in enumerate(selected_client_ids):
        optimizer = optim.Adam(
            all_client_models[client_id].parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        if args.dataset == "celeba":
            # Use the user ID to get the correct dataset
            user_id = get_user_id_by_idx(client_id, "data/celeba/train/by_user")
            train_loader = DataLoader(
                CelebA(
                    root="./data",
                    train=True,
                    user_id=user_id,
                    augmentation=DATA_AUGMENTATION,
                ),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
            )
        elif args.dataset == "femnist":
            # Use the user ID to get the correct dataset
            user_id = (get_user_id_by_idx(client_id, "data/femnist/train")).replace(
                ".json", ""
            )
            train_loader = DataLoader(
                FEMNIST(
                    root="./data",
                    train=True,
                    user_id=user_id,
                    resize=(args.model == "cnn"),
                    augmentation=DATA_AUGMENTATION,
                ),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
            )
        else:
            train_loader = train_loaders[client_id]

        # Train the client model
        local_train_loss = train(
            model=all_client_models[client_id],
            optimizer=optimizer,
            criterion=criterion,
            dataloader=train_loader,
            device=device,
            epochs=args.epochs,
            verbose=False,
        )

        # Evaluate the client model
        local_evaluation_result_on_train_data = evaluate_acc(
            model=all_client_models[client_id],
            dataloader=train_loader,
            device=device,
            class_wise=True,
        )

        # Store the evaluation results of the client for later printing
        client_evaluation_results[client_id] = {
            "model_size": calculate_model_size(
                all_client_models[client_id], print_result=False, unit="MB"
            ),
            "train_loss": local_train_loss,
            "train_acc": local_evaluation_result_on_train_data["accuracy"],
        }

    local_training_time = time()
    print(
        f"Local training time: {(local_training_time - distribution_time) / 60:.2f} min"
    )

    # <---------------------------------------- Aggregation ---------------------------------------->
    if args.method in ["fedavg"]:
        server.step(
            local_state_dicts=[
                model.state_dict() for model in all_client_models.values()
            ],
            selected_client_ids=selected_client_ids,
        )
    elif args.method in ["fedrolex", "heterofl", "fd", "fedecover"]:
        server.step(
            local_state_dicts=[
                model.state_dict() for model in all_client_models.values()
            ],
            selected_client_ids=selected_client_ids,
            submodel_param_indices_dicts=selected_submodel_param_indices_dicts,
        )
    else:
        raise NotImplementedError

    aggregation_time = time()
    print(f"Aggregation time: {(aggregation_time - local_training_time) / 60:.2f} min")

    # <---------------------------------------- Global Model Evaluation ---------------------------------------->
    global_evaluation_result = evaluate_acc(
        model=server.global_model,
        dataloader=global_test_loader,
        device=device,
        class_wise=True,
    )

    global_evaluation_time = time()
    print(
        f"Global model evaluation time: {(global_evaluation_time - aggregation_time) / 60:.2f} min"
    )

    # <---------------------------------------- Print Results ---------------------------------------->
    # Print client evaluation results
    for client_id in selected_client_ids:
        print(
            f"Client {client_id}\t"
            f"Capacity: {client_capacities[client_id]:.2f}\t"
            f"Model Size: {client_evaluation_results[client_id]['model_size']:.2f} MB\t"
            f"Train Loss: {client_evaluation_results[client_id]['train_loss']:.4f}\t"
            f"Train Acc: {client_evaluation_results[client_id]['train_acc']:.4f}\t"
        )

    # Print aggregated evaluation results
    print(
        "Aggregated\n"
        f"Global Test Loss: {global_evaluation_result['loss']:.4f}\t"
        f"Global Test Acc: {global_evaluation_result['accuracy']:.4f}"
    )

    print_results_time = time()
    print(
        f"Print results time: {(print_results_time - global_evaluation_time) / 60:.2f} min"
    )

    # Estimate ETA
    round_end_time = time()
    round_use_time = round_end_time - round_start_time
    print(
        f"Round {round + 1} use time: {round_use_time / 60:.2f} min, ETA: {(args.rounds - round - 1) * round_use_time / 3600:.2f} hours"
    )
    print("=" * 80)

end_time = time()
print(f"Total use time: {(end_time - start_time) / 3600:.2f} hours")
