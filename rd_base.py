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
    aggregate_shallow_resnet,
    recover_global_from_pruned_cnn,
    recover_global_from_pruned_resnet18,
    recover_global_from_pruned_shallow_resnet,
    vanilla_federated_averaging,
)
from modules.args_parser import get_args
from modules.constants import NORMALIZATION_STATS
from modules.data import create_non_iid_data
from modules.models import CNN, ShallowResNet
from modules.pruning import (
    prune_cnn,
    # prune_cnn_v2,
    prune_resnet18,
    prune_shallow_resnet,
    pruned_indices_dict_bagging_cnn,
    pruned_indices_dict_bagging_resnet18,
)
from modules.utils import (
    calculate_model_size,
    evaluate_acc,
    replace_bn_with_ln,
    test,
    train,
)

args = get_args()
SAVE_DIR = args.save_dir
MODEL_TYPE = args.model
DATASET = args.dataset
DATASET = args.dataset
if DATASET == "cifar10":
    NUM_CLASSES = 10
elif DATASET == "cifar100":
    NUM_CLASSES = 100
else:
    raise ValueError(f"Dataset {DATASET} not supported.")
LR_DECAY = args.lr_decay
ROUNDS = args.round
EPOCHS = args.epochs
LR = args.lr
BATCH_SIZE = args.batch_size
NUM_CLIENTS = args.num_clients
AGG_WAY = args.aggregation
DEBUGGING = args.debugging

# Set random seed for reproducibility
seed = 18
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Data preparation
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


# Model preparation
if MODEL_TYPE == "cnn":
    global_model = CNN(num_classes=NUM_CLASSES)
elif MODEL_TYPE == "resnet":
    global_model = resnet18(weights=None, num_classes=NUM_CLASSES)
    replace_bn_with_ln(global_model)
elif MODEL_TYPE == "shallow_resnet":
    global_model = ShallowResNet(num_classes=NUM_CLASSES)
else:
    raise ValueError(f"Model type {MODEL_TYPE} not supported.")
print(f"[Model Architecture]\n{global_model}")

num_models = NUM_CLIENTS
if num_models == 10:
    dropout_rates = [0.2, 0.2, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 0.8, 0.8]
    # dropout_rates = [0.1, 0.1, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 0.9, 0.9]
elif num_models == 20:
    dropout_rates = [0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
assert (
    len(dropout_rates) == num_models
), f"Length of dropout rates {len(dropout_rates)} does not match number of clients {num_models}."
learning_rates = [LR] * num_models
decay_rounds = [20, 50, 100]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()


results_class = []
train_loss_results = []
test_loss_results = []
test_acc_results = []


pruned_indices_dict_bags_for_each_p = [[] for _ in range(num_models)]


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

    if MODEL_TYPE == "cnn":
        pruned_indices_dicts = []
        for i in range(num_models):
            if len(pruned_indices_dict_bags_for_each_p[i]) == 0:
                pruned_indices_dict_bags_for_each_p[i] = (
                    pruned_indices_dict_bagging_cnn(dropout_rates[i])
                )
            optional_indices_dict = pruned_indices_dict_bags_for_each_p[i].pop()
            client_model, pruned_indices_dict = prune_cnn(
                global_model,
                dropout_rates[i],
                scaling=True,
                optional_indices_dict=optional_indices_dict,
            )
            all_client_models.append(client_model)
            pruned_indices_dicts.append(pruned_indices_dict)

        # all_client_models, pruned_indices_dicts = prune_cnn_groups_nested(
        #     global_model, dropout_rates
        # )
    elif MODEL_TYPE == "resnet":
        pruned_indices_dicts = []
        for i in range(num_models):
            if len(pruned_indices_dict_bags_for_each_p[i]) == 0:
                pruned_indices_dict_bags_for_each_p[i] = (
                    pruned_indices_dict_bagging_resnet18(dropout_rates[i])
                )
            optional_indices_dict = pruned_indices_dict_bags_for_each_p[i].pop()
            client_model, pruned_indices_dict = prune_resnet18(
                global_model,
                dropout_rates[i],
                optional_indices_dict=optional_indices_dict,
            )
            all_client_models.append(client_model)
            pruned_indices_dicts.append(pruned_indices_dict)

        # all_client_models, pruned_indices_dicts = prune_resnet18_groups_nested(
        #     global_model, dropout_rates
        # )
    elif MODEL_TYPE == "shallow_resnet":
        pruned_indices_dicts = []
        for i in range(num_models):
            client_model, pruned_indices_dict = prune_shallow_resnet(
                global_model, dropout_rates[i]
            )
            all_client_models.append(client_model)
            pruned_indices_dicts.append(pruned_indices_dict)
    else:
        raise ValueError(f"Model type {MODEL_TYPE} not supported.")

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
            local_model, criterion, test_loader, device=device, num_classes=NUM_CLASSES
        )
        model_size = calculate_model_size(local_model, print_result=False, unit="MB")
        print(
            f"Subset {i + 1}\tModel Size: {model_size:.2f} MB\tTrain Loss: {local_train_loss:.4f}\tTest Loss: {local_test_loss:.4f}\tTest Acc: {local_test_acc:.4f}"
        )
        round_train_loss_results[f"Subset {i + 1}"] = local_train_loss
        round_test_loss_results[f"Subset {i + 1}"] = local_test_loss
        round_test_acc_results[f"Subset {i + 1}"] = local_test_acc
        round_results_class[f"Subset {i + 1}"] = local_class_acc

    # debugging
    if DEBUGGING:
        for debug_i in range(num_models):
            debug_model = all_client_models[debug_i]
            debug_model_result = evaluate_acc(debug_model, test_loader, device=device)
            print(
                f"Subset {debug_i + 1}\tTest Loss: {debug_model_result["loss"]:.4f}\tTest Acc: {debug_model_result["accuracy"]:.4f}"
            )

            for param in global_model.parameters():
                param.data.zero_()

            evaluate_acc(global_model, test_loader, device=device)

            # for name, param in global_model.named_parameters():
            #     print(f"Parameter {name}: {param.data}")

            if MODEL_TYPE == "cnn":
                recovered_model = recover_global_from_pruned_cnn(
                    global_model,
                    debug_model,
                    pruned_indices_dicts[debug_i],
                    # scaler=(1 / (1 - dropout_rates[debug_i])) ** 2,
                )
            elif MODEL_TYPE == "resnet":
                recovered_model = recover_global_from_pruned_resnet18(
                    global_model,
                    debug_model,
                    pruned_indices_dicts[debug_i],
                    # scaler=(1 / (1 - dropout_rates[debug_i])) ** 2,
                )
            elif MODEL_TYPE == "shallow_resnet":
                recovered_model = recover_global_from_pruned_shallow_resnet(
                    global_model,
                    debug_model,
                    pruned_indices_dicts[debug_i],
                    # scaler=(1 / (1 - dropout_rates[debug_i])) ** 2,
                )
            recovered_model_result = evaluate_acc(
                recovered_model, test_loader, device=device
            )
            print(
                f"Subset {debug_i + 1} Rec\tTest Loss: {recovered_model_result['loss']:.4f}\tTest Acc: {recovered_model_result['accuracy']:.4f}"
            )

    # Aggregation
    if MODEL_TYPE == "cnn":
        if AGG_WAY == "sparse":
            aggregate_cnn(
                global_model,
                all_client_models,
                subset_sizes,
                indices_to_prune_conv1=[
                    pruned_indices_dicts[i]["layer1"] for i in range(num_models)
                ],
                indices_to_prune_conv2=[
                    pruned_indices_dicts[i]["layer2"] for i in range(num_models)
                ],
                indices_to_prune_conv3=[
                    pruned_indices_dicts[i]["layer3"] for i in range(num_models)
                ],
                indices_to_prune_fc=[
                    pruned_indices_dicts[i]["fc"] for i in range(num_models)
                ],
            )
        elif AGG_WAY == "recovery":
            aggregated_weight = vanilla_federated_averaging(
                models=[
                    recover_global_from_pruned_cnn(
                        global_model, all_client_models[i], pruned_indices_dicts[i]
                    )
                    for i in range(num_models)
                ],
                sample_numbers=subset_sizes,
            )
            global_model.load_state_dict(aggregated_weight)
    elif MODEL_TYPE == "resnet":
        if AGG_WAY == "sparse":
            aggregate_resnet18(
                global_model=global_model,
                local_models=all_client_models,
                client_weights=subset_sizes,
                pruned_indices_dicts=pruned_indices_dicts,
            )
        elif AGG_WAY == "recovery":
            aggregated_weight = vanilla_federated_averaging(
                models=[
                    recover_global_from_pruned_resnet18(
                        global_model, all_client_models[i], pruned_indices_dicts[i]
                    )
                    for i in range(num_models)
                ],
                sample_numbers=subset_sizes,
            )
            global_model.load_state_dict(aggregated_weight)
    elif MODEL_TYPE == "shallow_resnet":
        aggregate_shallow_resnet(
            global_model=global_model,
            local_models=all_client_models,
            client_weights=subset_sizes,
            pruned_indices_dicts=pruned_indices_dicts,
        )
    else:
        raise ValueError(f"Model type {MODEL_TYPE} not supported.")

    global_test_loss, global_test_acc, global_class_acc = test(
        global_model, criterion, test_loader, device=device, num_classes=NUM_CLASSES
    )
    print(
        f"Aggregated Test Loss: {global_test_loss:.4f}\tAggregated Test Acc: {global_test_acc:.4f}"
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

if SAVE_DIR is not None:
    with open(
        os.path.join(SAVE_DIR, f"rd_base_{MODEL_TYPE}_{DATASET}_train_loss.csv"), "w"
    ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=train_loss_results[0].keys())
        writer.writeheader()
        writer.writerows(train_loss_results)

    with open(
        os.path.join(SAVE_DIR, f"rd_base_{MODEL_TYPE}_{DATASET}_test_loss.csv"), "w"
    ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=test_loss_results[0].keys())
        writer.writeheader()
        writer.writerows(test_loss_results)

    with open(
        os.path.join(SAVE_DIR, f"rd_base_{MODEL_TYPE}_{DATASET}_test_acc.csv"), "w"
    ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=test_acc_results[0].keys())
        writer.writeheader()
        writer.writerows(test_acc_results)

    with open(
        os.path.join(SAVE_DIR, f"rd_base_{MODEL_TYPE}_{DATASET}_class_acc.pkl"), "wb"
    ) as f:
        pickle.dump(results_class, f)

    print("Results saved.")
else:
    print("Results not saved.")
