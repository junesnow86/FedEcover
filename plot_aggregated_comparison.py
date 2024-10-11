import matplotlib.pyplot as plt
import pandas as pd

DATE = "1009"
models = ["cnn"]
datasets = ["cifar10", "cifar100"]
distributions = ["iid", "alpha0.5", "alpha0.1"]
methods = ["fedavg", "heterofl", "fedrolex", "fedrd", "fedrame", "fedrame_dynamic_eta"]
plot_type = "acc"
axhline = "avg"
avg_line_start_round = 200

for model in models:
    for dataset in datasets:
        for distribution in distributions:
            file_paths = [
                f"results/{DATE}/{method}_{model}_{dataset}_{distribution}_aggregated_test_results.csv" for method in methods
            ]


            # Initialize the plot
            plt.figure(figsize=(10, 6))

            # Loop through each file path
            for i, method in enumerate(methods):
            # for i, method in enumerate(correction_coefficient):
                # Read the CSV file
                file_path = file_paths[i]
                df = pd.read_csv(file_path)

                # Extract the Round and Aggregated Test Acc columns
                rounds = df["Round"]
                if plot_type == "acc":
                    acc = df["Aggregated Test Acc"].iloc[:len(rounds)]
                    plt.plot(rounds, acc, label=f"{method}")
                    if axhline == "max":
                        max_acc = acc.max()
                        plt.axhline(
                            y=max_acc, color=f"C{i}", linestyle="--", label=f"{method} Max: {max_acc:.4f}"
                        )
                    elif axhline == "avg":
                        avg_acc = acc[rounds > 150].mean()
                        plt.axhline(
                            y=avg_acc, color=f"C{i}", linestyle="--", label=f"{method} Avg: {avg_acc:.4f}"
                        )
                elif plot_type == "loss":
                    loss = df["Aggregated Test Loss"]
                    plt.plot(rounds, loss, label=f"{method}")
                    min_loss = loss.min()
                    plt.axhline(
                        y=min_loss, color=f"C{i}", linestyle="--", label=f"{method} Min: {min_loss:.4f}"
                    )

            # Add labels and title
            plt.legend()
            plt.xlabel("Round")
            if plot_type == "acc":
                plt.ylabel("Aggregated Test Acc")
                plt.title(f"Aggregated Test Accuracy vs Round for Different Methods ({model}, {dataset}, {distribution})")
                plt.savefig(f"figures/{DATE}/aggregated_acc_comparison_{axhline}_{model}_{dataset}_{distribution}.png")
            elif plot_type == "loss":
                plt.ylabel("Aggregated Test Loss")
                plt.title(f"Aggregated Test Loss vs Round for Different Methods ({model}, {dataset}, {distribution})")
                plt.savefig(f"figures/{DATE}/aggregated_loss_comparison_{axhline}_{model}_{dataset}_{distribution}.png")

