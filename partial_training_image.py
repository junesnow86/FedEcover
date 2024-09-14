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

from modules.aggregation import (
    aggregate_cnn,
    aggregate_resnet18,
    vanilla_federated_averaging,
)
from modules.aggregation.aggregate_models import (
    recover_global_from_pruned_cnn,
    recover_global_from_pruned_resnet18,
)
from modules.args_parser import get_args
from modules.constants import NORMALIZATION_STATS
from modules.data import create_non_iid_data
from modules.debugging import replace_bn_with_identity
from modules.evaluation import evaluate_acc
from modules.models import CNN
from modules.pruning import (
    generate_model_pruned_indices_dicts_bag_for_cnn,
    generate_model_pruned_indices_dicts_bag_for_resnet18,
    prune_cnn,
    prune_resnet18,
)
from modules.training import train
from modules.utils import (
    calculate_model_size,
    replace_bn_with_ln,
)

# <==================== Parse arguments ====================>
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

# Set random seed for reproducibility
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# <==================== Data preparation ====================>
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

# exit()


# <==================== Model preparation ====================>
if MODEL_TYPE == "cnn":
    global_model = CNN(num_classes=NUM_CLASSES)
elif MODEL_TYPE == "resnet":
    global_model = resnet18(weights=None, num_classes=NUM_CLASSES)
    replace_bn_with_ln(global_model)
    if DEBUGGING:
        replace_bn_with_identity(global_model)
else:
    raise ValueError(f"Model type {MODEL_TYPE} not supported.")
print(f"[Model Architecture]\n{global_model}")
print(
    f"Global Model size: {calculate_model_size(global_model, print_result=False, unit='MB'):.2f} MB"
)

num_models = NUM_CLIENTS
optional_dropout_rates = [0.1, 0.25, 0.5, 0.75, 0.9]
weights = [0.05, 0.1, 0.15, 0.2, 0.5]
# optional_dropout_rates = [0.2, 0.5, 0.8]
# weights = [0.2, 0.3, 0.5]
client_dropout_rates = [
    random.choices(optional_dropout_rates, weights=weights)[0]
    for _ in range(num_models)
]
print(f"Client dropout rates: {client_dropout_rates}")
dropout_rate_count = {}
for rate in client_dropout_rates:
    if rate in dropout_rate_count:
        dropout_rate_count[rate] += 1
    else:
        dropout_rate_count[rate] = 1
print(f"Client dropout rate count: {dropout_rate_count}")
optional_model_pruned_indices_dicts = {p: [] for p in optional_dropout_rates}


# <==================== Training, aggregation and evluation ====================>
train_loss_results = []
test_loss_results = []
test_acc_results = []
class_wise_results = []
local_validation_results = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

start_time = time()
for round in range(ROUNDS):
    round_start_time = time()
    print(f"Round {round + 1}")

    round_train_loss_results = {"Round": round + 1}
    round_test_loss_results = {"Round": round + 1}
    round_test_acc_results = {"Round": round + 1}
    round_class_wise_results = {"Round": round + 1}

    # Radomly select a ratio of clients to participate in this round
    selected_client_ids = np.random.choice(
        range(num_models), int(num_models * SELECT_RATIO), replace=False
    )
    print(f"Selected {len(selected_client_ids)} client IDs: {selected_client_ids}")

    all_client_models = {}
    model_pruned_indices_dicts = {}

    # <-------------------- Pruning -------------------->
    if MODEL_TYPE == "cnn":
        if METHOD == "bagging-rd":
            for client_id in selected_client_ids:
                if (
                    len(
                        optional_model_pruned_indices_dicts[
                            client_dropout_rates[client_id]
                        ]
                    )
                    == 0
                ):
                    optional_model_pruned_indices_dicts[
                        client_dropout_rates[client_id]
                    ] = generate_model_pruned_indices_dicts_bag_for_cnn(
                        client_dropout_rates[client_id]
                    )

                optional_indices_dict = optional_model_pruned_indices_dicts[
                    client_dropout_rates[client_id]
                ].pop()
                client_model, model_pruned_indices_dict = prune_cnn(
                    global_model,
                    client_dropout_rates[client_id],
                    optional_indices_dict=optional_indices_dict,
                )
                all_client_models[client_id] = client_model
                model_pruned_indices_dicts[client_id] = model_pruned_indices_dict
        else:
            raise NotImplementedError
    elif MODEL_TYPE == "resnet":
        if METHOD == "bagging-rd":
            for client_id in selected_client_ids:
                if (
                    len(
                        optional_model_pruned_indices_dicts[
                            client_dropout_rates[client_id]
                        ]
                    )
                    == 0
                ):
                    optional_model_pruned_indices_dicts[
                        client_dropout_rates[client_id]
                    ] = generate_model_pruned_indices_dicts_bag_for_resnet18(
                        client_dropout_rates[client_id]
                    )

                optional_indices_dict = optional_model_pruned_indices_dicts[
                    client_dropout_rates[client_id]
                ].pop()
                client_model, model_pruned_indices_dict = prune_resnet18(
                    global_model,
                    client_dropout_rates[client_id],
                    optional_indices_dict=optional_indices_dict,
                )
                all_client_models[client_id] = client_model
                model_pruned_indices_dicts[client_id] = model_pruned_indices_dict
        else:
            raise NotImplementedError
    else:
        raise ValueError(f"Model type {MODEL_TYPE} not supported.")

    # <-------------------- Local Training -------------------->
    for client_id in selected_client_ids:
        # optimizer = optim.SGD(
        #     all_client_models[client_id].parameters(),
        #     lr=LR,
        #     momentum=0.9,
        #     weight_decay=5e-4,
        # )
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

        local_test_loss = local_evaluation_result["loss"]
        local_test_acc = local_evaluation_result["accuracy"]
        local_class_acc = local_evaluation_result["class_wise_accuracy"]
        model_size = calculate_model_size(
            all_client_models[client_id], print_result=False, unit="MB"
        )
        print(
            f"Client {client_id}\tDropout Rate: {client_dropout_rates[client_id]}\tModel Size: {model_size:.2f} MB\tTrain Loss: {local_train_loss:.4f}\tTrain Acc: {local_evaluation_result_on_train_data['accuracy']:.4f}\tTest Loss: {local_test_loss:.4f}\tTest Acc: {local_test_acc:.4f}"
        )

        round_train_loss_results[f"Client {client_id}"] = local_train_loss
        round_test_loss_results[f"Client {client_id}"] = local_test_loss
        round_test_acc_results[f"Client {client_id}"] = local_test_acc
        round_class_wise_results[f"Client {client_id}"] = local_class_acc

    # <-------------------- Aggregation -------------------->
    client_weights = [subset_sizes[i] for i in selected_client_ids]
    if MODEL_TYPE == "cnn":
        if AGG_WAY == "sparse":
            aggregate_cnn(
                global_model=global_model,
                local_models=[all_client_models[i] for i in selected_client_ids],
                model_pruned_indices_dicts=[
                    model_pruned_indices_dicts[i] for i in selected_client_ids
                ],
                client_weights=client_weights,
            )
        elif AGG_WAY == "recovery":
            aggregated_weight = vanilla_federated_averaging(
                models=[
                    recover_global_from_pruned_cnn(
                        global_model,
                        all_client_models[i],
                        model_pruned_indices_dicts[i],
                    )
                    for i in range(num_models)
                ],
                client_weights=client_weights,
            )
            global_model.load_state_dict(aggregated_weight)
    elif MODEL_TYPE == "resnet":
        if AGG_WAY == "sparse":
            aggregate_resnet18(
                global_model=global_model,
                local_models=[all_client_models[i] for i in selected_client_ids],
                model_pruned_indices_dicts=[
                    model_pruned_indices_dicts[i] for i in selected_client_ids
                ],
                client_weights=client_weights,
            )
        elif AGG_WAY == "recovery":
            aggregated_weight = vanilla_federated_averaging(
                models=[
                    recover_global_from_pruned_resnet18(
                        global_model,
                        all_client_models[i],
                        model_pruned_indices_dicts[i],
                    )
                    for i in range(num_models)
                ],
                client_weights=client_weights,
            )
            global_model.load_state_dict(aggregated_weight)
    else:
        raise ValueError(f"Model type {MODEL_TYPE} not supported.")

    # Evaluate the global model
    global_evaluation_result = evaluate_acc(
        model=global_model,
        dataloader=global_test_loader,
        device=device,
        class_wise=True,
    )
    global_test_loss = global_evaluation_result["loss"]
    global_test_acc = global_evaluation_result["accuracy"]
    global_class_acc = global_evaluation_result["class_wise_accuracy"]
    print(
        f"Aggregated Test Loss: {global_test_loss:.4f}\tAggregated Test Acc: {global_test_acc:.4f}"
    )

    round_test_loss_results["Aggregated"] = global_test_loss
    round_test_acc_results["Aggregated"] = global_test_acc
    round_class_wise_results["Aggregated"] = global_class_acc

    # Collect round results
    train_loss_results.append(round_train_loss_results)
    test_loss_results.append(round_test_loss_results)
    test_acc_results.append(round_test_acc_results)
    class_wise_results.append(round_class_wise_results)

    if (round + 1) % LOCAL_VALIDATION_FREQUENCY == 0 and LOCAL_TRAIN_RATIO < 1.0:
        # Evaluate the global model on **all** local validation sets
        round_local_validation_results = {"Round": round + 1}
        avg_local_validation_acc = 0.0
        dropout_rate_results = {p: [] for p in optional_dropout_rates}
        for i in range(num_models):
            local_validation_result = evaluate_acc(
                model=global_model,
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
        avg_local_validation_acc /= num_models
        round_local_validation_results["Average"] = avg_local_validation_acc
        for p in optional_dropout_rates:
            dropout_rate_results[p] = np.mean(dropout_rate_results[p])
            round_local_validation_results[f"Drpout Rate {p}"] = dropout_rate_results[p]
        local_validation_results.append(round_local_validation_results)
        # Print local validation results
        for key, value in round_local_validation_results.items():
            if key != "Round":
                print(f"{key}: {value:.4f} | {value - avg_local_validation_acc:.4f} (compared to average)")

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
        os.path.join(SAVE_DIR, f"{METHOD}_{MODEL_TYPE}_{DATASET}_train_loss.csv"), "w"
    ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=train_loss_results[0].keys())
        writer.writeheader()
        writer.writerows(train_loss_results)

    with open(
        os.path.join(SAVE_DIR, f"{METHOD}_{MODEL_TYPE}_{DATASET}_test_loss.csv"), "w"
    ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=test_loss_results[0].keys())
        writer.writeheader()
        writer.writerows(test_loss_results)

    with open(
        os.path.join(SAVE_DIR, f"{METHOD}_{MODEL_TYPE}_{DATASET}_test_acc.csv"), "w"
    ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=test_acc_results[0].keys())
        writer.writeheader()
        writer.writerows(test_acc_results)

    with open(
        os.path.join(SAVE_DIR, f"{METHOD}_{MODEL_TYPE}_{DATASET}_class_acc.pkl"), "wb"
    ) as f:
        pickle.dump(class_wise_results, f)

    with open(
        os.path.join(SAVE_DIR, f"{METHOD}_{MODEL_TYPE}_{DATASET}_local_validation.csv"),
        "w",
    ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=local_validation_results[0].keys())
        writer.writeheader()
        writer.writerows(local_validation_results)

    print("Results saved.")
else:
    print("Results not saved.")
