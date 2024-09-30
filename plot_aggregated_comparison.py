import matplotlib.pyplot as plt
import pandas as pd


MODEL = "cnn"
DATASET = "cifar100"
DATE = "0928"
DISTRIBUTION = "alpha0.5"

# List of file paths for the 6 CSV files
# methods = ["fedavg", "heterofl", "fedrd", "fedrolex", "rdbagging_frequent", "fedavg_largest"]
# methods = ["fedavg", "fedrd", "legacy", "rdbagging", "fedavg_largest"]
# methods = ["fedavg", "legacy", "fedavg_largest"]
# methods = ["fedrd", "legacy", "rdbagging", "rdbagging_client", "rdbagging_frequent"]
# methods = ["rdbagging_frequent"]
# methods = ["fedavg", "heterofl", "fedrd", "fedrolex", "rdbagging_steady", "rdbagging_client", "rdbagging_frequent"]
methods = ["fedavg", "heterofl", "fedrd", "fedrolex", "rdbagging_frequent", "fedrame", "fedrame_momentum0.1", "fedrame_momentum0.5"]
file_paths = [
    f"results/{DATE}/{method}_{MODEL}_{DATASET}_{DISTRIBUTION}_aggregated_test_results.csv" for method in methods
    # f"results/0924/{method}_{MODEL}_{DATASET}_aggregated_test_results.csv" for method in methods
    # f"results/0925/{method}_{MODEL}_{DATASET}_aggregated_test_results.csv" for method in methods
    # f"results/0919/{method}_aggregated_test_results.csv" for method in methods
]

# method = "fedrame"
# correction_coefficient = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, "non-correction"]
# file_paths = [
#     f"results/{DATE}/{method}_{MODEL}_{DATASET}_{DISTRIBUTION}_{correction}_aggregated_test_results.csv" for correction in correction_coefficient
# ]

plot_type = "acc"
axhline = "max"

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
    plt.title(f"Aggregated Test Accuracy vs Round for Different Methods ({MODEL}, {DATASET}, {DISTRIBUTION})")
    plt.savefig(f"figures/{DATE}/aggregated_acc_comparison_{axhline}_{MODEL}_{DATASET}_{DISTRIBUTION}.png")
elif plot_type == "loss":
    plt.ylabel("Aggregated Test Loss")
    plt.title(f"Aggregated Test Loss vs Round for Different Methods ({MODEL}, {DATASET}, {DISTRIBUTION})")
    plt.savefig(f"figures/{DATE}/aggregated_loss_comparison_{axhline}_{MODEL}_{DATASET}_{DISTRIBUTION}.png")

