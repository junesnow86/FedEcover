import os
import random
from time import time

import numpy as np
import torch

from modules.args_parser import get_args
from modules.constants import OPTIONAL_CLIENT_CAPACITY_DISTRIBUTIONS
from modules.models import CNN, FEMNISTCNN, custom_resnet18
from modules.servers import (
    ServerFD,
    ServerFedEcover,
    ServerFedRolex,
    ServerHomo,
    ServerStatic,
)
from modules.utils import (
    calculate_model_size,
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

if args.data_augmentation:
    if "cifar" in args.dataset and NUM_CLIENTS != 100:
        DATA_AUGMENTATION = False
    else:
        DATA_AUGMENTATION = True
else:
    DATA_AUGMENTATION = False


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
decay_steps = [i for i in range(args.Tdi, args.Tds + 1, args.Tdi)]
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
print("\n" + "=" * 80 + "Basic Information" + "=" * 80 + "\n")
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

print("\n" + "=" * 80 + "Local Training Information" + "=" * 80 + "\n")
print(f"Number of local epochs: {args.epochs}")
print(f"Batch size of local training: {args.batch_size}")
print(f"Learning rate of local training: {args.lr}")
print(f"Weight decay of local training: {args.weight_decay}")
print(f"Whether use data augmentation: {DATA_AUGMENTATION}")
print(f"Number of dataloader workers: {args.num_workers}")

print("\n" + "=" * 80 + "Aggregation Information" + "=" * 80 + "\n")
if args.model == "resnet":
    print(f"Model normalization layer type: {args.norm_type}")
print(f"Global step-size: {args.eta_g}")
print(f"Whether use global step-size decay: {args.global_lr_decay}")
print(f"Gamma: {args.gamma}")
print(f"GSD stop round: {args.Tds}")
print(f"GSD decay interval: {args.Tdi}")

print("\n" + "=" * 80 + "Evaluation Information" + "=" * 80 + "\n")


# <======================================== Training, aggregation and evaluation ========================================>
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

    if args.method == "fedrolex":
        server.roll_indices()

    print(
        f"{len(selected_client_ids)} are selected, specific IDs: {selected_client_ids}"
    )

    distribution_time = time()

    # Estimate ETA
    round_end_time = time()
    round_use_time = round_end_time - round_start_time

    print(f"Submodel distribution time: {distribution_time - round_start_time:.2f} sec")
    print(
        f"Round {round + 1} use time: {round_use_time / 60:.2f} min, ETA: {(args.rounds - round - 1) * round_use_time / 3600:.2f} hours"
    )
    print("-" * 80)

end_time = time()
print(f"Total use time: {(end_time - start_time) / 3600:.2f} hours")

# Print the final neuron selection count
print("\n" + "=" * 80 + "Neuron Selection Count" + "=" * 80 + "\n")
print(server.neuron_selection_count)

save_dir = f"figures/param-selection-count/{args.method}-{args.model}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

y_min, y_max = None, None
if args.method == "fedecover":
    y_min, y_max = -10, 10
#     elif args.model == "femnistcnn":
#         y_min = 2500
# elif args.method == "fd":
#     if args.model == "cnn":
#         y_min, y_max = 0.9, 1.0
#     elif args.model == "femnistcnn":
#         y_min = 2500
# elif args.method == "fedrolex":
#     if args.model == "cnn":
#         y_min, y_max = 0.7, 1.0
server.visualize_neuron_selection_count(save_dir, y_min=y_min, y_max=y_max, type="bar")
server.save_neuron_selection_count(
    f"/home/ljt/research/FedEcover/results/param-selection-count/{args.method}.npy"
)
