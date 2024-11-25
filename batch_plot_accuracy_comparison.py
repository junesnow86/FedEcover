import os

import matplotlib.pyplot as plt
import pandas as pd

# date = "1108-10clients"
# date = "1112-100clients"
# date = "1123-alpha-10clients"
date = "1123-alpha-100clients"

models = ["cnn"]
# models = ["resnet"]

# datasets = ["cifar10"]
datasets = ["cifar100"]
# datasets = ["cifar10", "cifar100"]
# datasets = ["tinyimagenet"]

distributions = ["alpha0.2"]
# distributions = ["alpha0.5"]
# distributions = ["alpha0.1", "alpha0.5", "alpha1.0", "alpha5.0"]

methods = ["fedavg", "heterofl", "fedrolex", "fedrd", "fedrame2"]
# methods = ["fedavg", "fedavg-no-gsd"]
# methods = ["heterofl", "heterofl-no-gsd"]
# methods = ["fedrolex", "fedrolex-no-gsd"]
# methods = ["fedrd", "fedrd-no-gsd"]
# methods = ["fedrame2", "fedrame2-no-gsd"]

# annotation = "fedavg"
# annotation = "heterofl"
# annotation = "fedrolex"
# annotation = "fedrd"
# annotation = "fedrame2"
annotation = None

# axhline = "avg"
# axhline = "max"
axhline = None

# capacity = "capacity2"
capacity = "capacity0"

# num_clients = 10
num_clients = 100

avg_line_start_round = 200
total_rounds = 300

method_colors = {
    "fedavg": "C0",
    "heterofl": "C1",
    "fedrolex": "C2",
    "fedrd": "C4",
    "fedrame": "C6",
    "fedrame2": "C3",
}

method_labels = {
    "fedavg": "FedAvg + GSD",
    "heterofl": "HeteroFL + GSD",
    "fedrolex": "FedRolex + GSD",
    "fedrd": "FD-m + GSD",
    "fedrame": "FedEcover0",
    "fedrame2": "FedEcover",
    "fedavg-no-gsd": "FedAvg w/o GSD",
    "heterofl-no-gsd": "HeteroFL w/o GSD",
    "fedrolex-no-gsd": "FedRolex w/o GSD",
    "fedrd-no-gsd": "FD-m w/o GSD",
    "fedrame-no-gsd": "FedEcover0 w/o GSD",
    "fedrame2-no-gsd": "FedEcover w/o GSD",
}

for model in models:
    for dataset in datasets:
        for distribution in distributions:
            print(
                f"Model: {model}, Dataset: {dataset}, Distribution: {distribution}, Capacity: {capacity}, Num Clients: {num_clients}"
            )

            # Initialize the plot
            # plt.figure(figsize=(8, 6))
            plt.figure()
            plt.grid(True)

            fedavg_mean_acc = None

            with_annotation = False
            for i, method in enumerate(methods):
                if "no-gsd" in method:
                    with_annotation = True
                    break

            for i, method in enumerate(methods):
                # Read the CSV file
                try:
                    file_path = f"results/{date}/{method}_{model}_{dataset}_{distribution}_{capacity}_{num_clients}clients.csv"
                    df = pd.read_csv(file_path)
                except FileNotFoundError:
                    print(f"File {file_path} not found.")
                    continue

                label = method_labels.get(method, method)
                if with_annotation:
                    if "no-gsd" not in method:
                        label = f"{label} w/ GSD"

                # Extract the Round and Aggregated Test Acc columns
                rounds = df["Round"].iloc[:total_rounds]
                acc = (
                    df["Aggregated Test Acc"].iloc[: len(rounds)] * 100
                )  # Convert to percentage

                # color = method_colors.get(method, "C5")
                color = method_colors.get(method, "black")
                plt.plot(rounds, acc, label=f"{label}", color=color)

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
                        # color=f"C{i}",
                        color=color,
                        linestyle="--",
                        label=f"{label} Avg: {avg_acc:.2f}%",
                    )

            # Add labels and title
            if axhline is None:
                plt.legend(fontsize=16)
                # plt.legend(bbox_to_anchor=(0.5, 1.01), loc="lower center", ncol=5, fontsize=11)
                # plt.legend(bbox_to_anchor=(0.5, 1.01), loc="lower center", ncol=5, fontsize=16)
            else:
                plt.legend(fontsize=11)
            plt.xlabel("Communication Round", fontsize=14)
            plt.ylabel("Top-1 Accuracy (%)", fontsize=14)

            plt.tick_params(axis="both", which="major", labelsize=12)  # Increase tick size
            plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)

            figure_dir = f"figures/{date}"
            if not os.path.exists(figure_dir):
                os.makedirs(figure_dir)

            if num_clients is not None:
                if annotation is not None:
                    fig_save_path = f"figures/{date}/{model}_{dataset}_{distribution}_{capacity}_{num_clients}clients_{axhline}_{total_rounds}rounds_{annotation}.png"
                else:
                    fig_save_path = f"figures/{date}/{model}_{dataset}_{distribution}_{capacity}_{num_clients}clients_{axhline}_{total_rounds}rounds.png"
            else:
                fig_save_path = f"figures/{date}/{model}_{dataset}_{distribution}_{capacity}_{axhline}_{total_rounds}rounds.png"
            plt.savefig(fig_save_path)
            print(f"Figure saved at {fig_save_path}")

            if fedavg_mean_acc is not None:
                fedavg_reach_round = None
                for method in methods:
                    file_path = f"results/{date}/{method}_{model}_{dataset}_{distribution}_{capacity}_{num_clients}clients.csv"
                    df = pd.read_csv(file_path)

                    label = method_labels.get(method, method)

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
