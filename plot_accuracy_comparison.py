import os

import matplotlib.pyplot as plt
import pandas as pd

model = "cnn"
dataset = "cifar100"
distribution = "alpha0.5"
capacity = "capacity0"
num_clients = "100clients"
methods = ["fedavg", "heterofl", "fedrolex", "fd", "fedecover"]

avg_line_start_round = 200  # Specify the round to start calculating the mean and std
total_rounds = 300

method_colors = {
    "fedavg": "C0",
    "heterofl": "C1",
    "fedrolex": "C2",
    "fd": "C4",
    "fedecover": "C3",
}

method_labels = {
    "fedavg": "FedAvg + GSD",
    "heterofl": "HeteroFL + GSD",
    "fedrolex": "FedRolex + GSD",
    "fd": "FD-m + GSD",
    "fedecover": "FedEcover",
    "fedavg-no-gsd": "FedAvg w/o GSD",
    "heterofl-no-gsd": "HeteroFL w/o GSD",
    "fedrolex-no-gsd": "FedRolex w/o GSD",
    "fd-no-gsd": "FD-m w/o GSD",
    "fedecover-no-gsd": "FedEcover w/o GSD",
}

print(
    f"Model: {model}, Dataset: {dataset}, Distribution: {distribution}, Capacity: {capacity}, Num Clients: {num_clients}"
)

plt.figure()
plt.grid(True)

fedavg_mean_acc = None

for i, method in enumerate(methods):
    # Read the CSV file
    try:
        file_path = f"results/{method}_{model}_{dataset}_{distribution}_{capacity}_{num_clients}.csv"
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        continue

    # Extract the Round and Aggregated Test Acc columns
    rounds = df["Round"].iloc[:total_rounds]
    acc = df["Aggregated Test Acc"].iloc[: len(rounds)] * 100  # Convert to percentage

    # Plot the accuracy curve for the method
    label = method_labels.get(method, method)
    color = method_colors.get(method, "black")
    plt.plot(rounds, acc, label=f"{label}", color=color)

    # Calculate mean and standard deviation after avg_line_start_round
    acc_after_avg_line_start_round = acc[rounds > avg_line_start_round]
    mean_acc = acc_after_avg_line_start_round.mean()
    std_acc = acc_after_avg_line_start_round.std()
    print(f"{label} Mean: {mean_acc:.2f}%, Std: {std_acc:.2f}%")

    if method == "fedavg":
        fedavg_mean_acc = mean_acc

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

plt.legend()
plt.xlabel("Communication Round", fontsize=14)
plt.ylabel("Top-1 Accuracy (%)", fontsize=14)

plt.tick_params(axis="both", which="major", labelsize=12)  # Increase tick size
plt.subplots_adjust(left=0.1, right=0.95, top=0.95)

figure_dir = "figures"
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

fig_save_path = f"figures/{model}_{dataset}_{distribution}_{capacity}_{num_clients}_{total_rounds}rounds.png"
plt.savefig(fig_save_path)
print(f"Figure saved at {fig_save_path}")
print("-" * 50)
