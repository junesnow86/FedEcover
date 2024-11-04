import os

import matplotlib.pyplot as plt
import pandas as pd

date = "1031"

models = ["cnn"]
# models = ["resnet"]

# datasets = ["cifar10"]
datasets = ["cifar100"]
# datasets = ["cifar10", "cifar100"]
# datasets = ["tinyimagenet"]

# distributions = ["iid", "alpha0.5", "alpha0.1"]
# distributions = ["alpha0.1"]
distributions = ["alpha0.5"]
# distributions = ["alpha1.0"]

# methods = ["fedavg", "heterofl", "fedrolex", "fedrd", "fedrame"]
# methods = ["fedavg", "heterofl", "fedrolex", "fedrd", "fedrame2"]
# methods = ["fedavg", "heterofl", "fedrolex", "fedrd", "fedrame", "fedrame2"]
# methods = ["fedavg", "heterofl", "fedrolex", "fedrame"]
# methods = ["fedavg", "heterofl", "fedrolex", "fedrd", "fedrame", "fedrame_dynamic_eta"]
# methods = ["heterofl", "fedrolex", "fedrd", "fedrame"]
# methods = ["fedavg", "fedrd", "fedrame"]
# methods = ["fedrd", "fedrame"]
# methods = ["fedrd", "fedrd-pooling-debugging"]
# methods = ["fedrd", "fedrd-crop"]
# methods = ["fedrd", "fedrd-noaug-debugging"]
# methods = ["fedrd", "fedrd-aug"]
# methods = ["fedrame", "fedrame-aug"]
# methods = ["fedavg", "fedrd", "fedrame"]
# methods = ["fedrd", "fedrd-20epochs"]
# methods = ["fedrame", "fedrame-20epochs"]
# methods = ["fedrame", "fedrame2"]
# methods = ["fedrd", "fedrd-20epochs", "fedrame", "fedrame-20epochs"]
# methods = ["fedavg", "fedavg-no-gsd"]
# methods = ["heterofl", "heterofl-no-gsd"]
# methods = ["fedrolex", "fedrolex-no-gsd"]
# methods = ["fedrd", "fedrd-no-gsd"]
methods = ["fedrame", "fedrame-no-gsd"]

annotation = "fedrame"

axhline = "avg"
# axhline = "max"
# axhline = None

capacity = "capacity0"
# capacity = "capacity2"

# num_clients = 10
num_clients = 100

avg_line_start_round = 200
total_rounds = 300


for model in models:
    for dataset in datasets:
        for distribution in distributions:
            # if num_clients is not None:
            #     file_paths = [
            #         f"results/{date}/{method}_{model}_{dataset}_{distribution}_{capacity}_{num_clients}clients.csv"
            #         for method in methods
            #     ]
            # else:
            #     file_paths = [
            #         f"results/{date}/{method}_{model}_{dataset}_{distribution}_{capacity}.csv"
            #         for method in methods
            #         # f"results/{date}/{method}_{model}_{dataset}_{distribution}_{capacity}_aggregated_test_results.csv" for method in methods
            #         # f"results/{date}/{method}_{model}_{dataset}_{distribution}_aggregated_test_results.csv" for method in methods
            #     ]
            print(
                f"Model: {model}, Dataset: {dataset}, Distribution: {distribution}, Capacity: {capacity}, Num Clients: {num_clients}"
            )

            # Initialize the plot
            plt.figure(figsize=(10, 6))

            fedavg_mean_acc = None

            for i, method in enumerate(methods):
                # Read the CSV file
                # file_path = file_paths[i]
                try:
                    file_path = f"results/{date}/{method}_{model}_{dataset}_{distribution}_{capacity}_{num_clients}clients.csv"
                    df = pd.read_csv(file_path)
                except FileNotFoundError:
                    print(f"File {file_path} not found.")
                    continue

                if method == "fedavg":
                    label = "FedAvg"
                elif method == "heterofl":
                    label = "HeteroFL"
                elif method == "fedrolex":
                    label = "FedRolex"
                elif method == "fedrd":
                    label = "Federated Dropout"
                elif method == "fedrame":
                    label = "FlexiCover"
                elif method == "fedrame2" and "fedrame" not in methods:
                    label = "FlexiCover"
                else:
                    label = method

                # Extract the Round and Aggregated Test Acc columns
                rounds = df["Round"].iloc[:total_rounds]
                acc = (
                    df["Aggregated Test Acc"].iloc[: len(rounds)] * 100
                )  # Convert to percentage

                plt.plot(rounds, acc, label=f"{label}")

                # Calculate mean and standard deviation after avg_line_start_round
                acc_after_avg_line_start_round = acc[rounds > avg_line_start_round]
                mean_acc = acc_after_avg_line_start_round.mean()
                std_acc = acc_after_avg_line_start_round.std()
                print(f"{label} Mean: {mean_acc:.2f}%, Std: {std_acc:.2f}%")

                if method == "fedavg":
                    fedavg_mean_acc = mean_acc

                if axhline == "max":
                    max_acc = acc.max()
                    plt.axhline(
                        y=max_acc,
                        color=f"C{i}",
                        linestyle="--",
                        label=f"{label} Max: {max_acc:.4f}",
                    )
                elif axhline == "avg":
                    avg_acc = acc[rounds > avg_line_start_round].mean()
                    plt.axhline(
                        y=avg_acc,
                        color=f"C{i}",
                        linestyle="--",
                        label=f"{label} Avg: {avg_acc:.2f}%",
                    )

            # Add labels and title
            if axhline is None:
                plt.legend(bbox_to_anchor=(0.5, 1.01), loc="lower center", ncol=5)
            else:
                plt.legend()
            plt.xlabel("Communication Round")
            plt.ylabel("Top-1 Accuracy (%)")
            # if axhline == "max":
            #     plt.title(
            #         f"Global Test Accuracy ({model}, {dataset}, {distribution}, {capacity}, {axhline})"
            #     )
            # elif axhline == "avg":
            #     plt.title(
            #         f"Global Test Accuracy ({model}, {dataset}, {distribution}, {capacity}, {axhline} start round {avg_line_start_round})"
            #     )
            # else:
            #     plt.title(
            #         f"Global Test Accuracy ({model}, {dataset}, {distribution}, {capacity})"
            #     )
            figure_dir = f"figures/{date}"
            if not os.path.exists(figure_dir):
                os.makedirs(figure_dir)

            if num_clients is not None:
                if annotation is not None:
                    plt.savefig(
                        f"figures/{date}/{model}_{dataset}_{distribution}_{capacity}_{num_clients}clients_{axhline}_{total_rounds}rounds_{annotation}.png"
                    )
                else:
                    plt.savefig(
                        f"figures/{date}/{model}_{dataset}_{distribution}_{capacity}_{num_clients}clients_{axhline}_{total_rounds}rounds.png"
                    )
            else:
                plt.savefig(
                    f"figures/{date}/{model}_{dataset}_{distribution}_{capacity}_{axhline}_{total_rounds}rounds.png"
                )
                # plt.savefig(f"figures/{date}/{model}_{dataset}_{distribution}_{axhline}.png")

            if fedavg_mean_acc is not None:
                fedavg_reach_round = None
                for method in methods:
                    file_path = f"results/{date}/{method}_{model}_{dataset}_{distribution}_{capacity}_{num_clients}clients.csv"
                    df = pd.read_csv(file_path)

                    if method == "fedavg":
                        label = "FedAvg"
                    elif method == "heterofl":
                        label = "HeteroFL"
                    elif method == "fedrolex":
                        label = "FedRolex"
                    elif method == "fedrd":
                        label = "Federated Dropout"
                    elif method == "fedrame":
                        label = "FlexiCover"
                    else:
                        label = method

                    # Extract the Round and Aggregated Test Acc columns
                    rounds = df["Round"].iloc[:total_rounds]
                    acc = df["Aggregated Test Acc"].iloc[: len(rounds)] * 100

                    try:
                        first_reach_round = rounds[acc >= fedavg_mean_acc].iloc[0]
                        if method == "fedavg":
                            fedavg_reach_round = first_reach_round
                        speedup = fedavg_reach_round / first_reach_round
                        print(
                            f"{label} first reaches FedAvg mean accuracy at round {first_reach_round}, Speedup: {speedup:.2f}"
                        )
                    except IndexError:
                        print(f"{label} never reachs FedAvg mean accuracy")

            print("-" * 50)
