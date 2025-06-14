import os
import matplotlib.pyplot as plt
import pandas as pd

# plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False

method_styles = {
    "fedavg": {"linestyle": "-", "marker": "o", "markevery": 20, "linewidth": 1, "markersize": 8},
    "heterofl": {"linestyle": "--", "marker": "s", "markevery": 20, "linewidth": 1, "markersize": 7},
    "fedrolex": {"linestyle": "-.", "marker": "^", "markevery": 20, "linewidth": 1, "markersize": 8},
    "fd": {"linestyle": ":", "marker": "D", "markevery": 20, "linewidth": 1, "markersize": 6},
    "fedecover": {"linestyle": "-", "marker": "*", "markevery": 20, "linewidth": 1, "markersize": 10},
    "fedavg-no-gsd": {"linestyle": "--", "marker": "o", "markevery": 20, "linewidth": 1.5, "markersize": 6},
    "heterofl-no-gsd": {"linestyle": "-.", "marker": "s", "markevery": 20, "linewidth": 1.5, "markersize": 6},
    "fedrolex-no-gsd": {"linestyle": ":", "marker": "^", "markevery": 20, "linewidth": 1.5, "markersize": 6},
    "fd-no-gsd": {"linestyle": "-", "marker": "D", "markevery": 20, "linewidth": 1.5, "markersize": 5},
    "fedecover-no-gsd": {"linestyle": "--", "marker": "*", "markevery": 20, "linewidth": 1.5, "markersize": 8},
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

avg_line_start_round = 200  # Specify the round to start calculating the mean and std
total_rounds = 300

methods = ["fedavg", "heterofl", "fedrolex", "fd", "fedecover"]
# methods = ["fedecover", "fedecover-no-gsd"]
# methods = ["fd", "fd-no-gsd"]
# methods = ["fedrolex", "fedrolex-no-gsd"]
# methods = ["heterofl", "heterofl-no-gsd"]
# methods = ["fedavg", "fedavg-no-gsd"]

model = "cnn"
dataset = "cifar100"
distribution = "alpha0.5"
capacity = "capacity2"
num_clients = "10clients"

# sub_dir = "param-sensitivity/Tds200-Tdi10"
# sub_dir = "param-sensitivity/Tds100-Tdi10"
# sub_dir = "param-sensitivity/no-gsd"
# sub_dir = "iid-gsd"
# sub_dir = "20250217"
# sub_dir = "femnist20250219"
sub_dir = "small-client-amount"
# sub_dir = "large-client-amount"
# sub_dir = "ablation"
csv_dir = "results"
fig_dir = "figures"
if sub_dir:
    csv_dir = os.path.join(csv_dir, sub_dir)
    fig_dir = os.path.join(fig_dir, sub_dir)
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

fig_save_path = os.path.join(
    fig_dir,
    f"{model}-{dataset}-{distribution}-{capacity}-{num_clients}-black.pdf",
)
# fig_save_path = os.path.join(fig_dir, "fedecover-ablation-10clients.pdf")
# fig_save_path = os.path.join(fig_dir, "fd-ablation-10clients.pdf")
# fig_save_path = os.path.join(fig_dir, "fedrolex-ablation-10clients.pdf")
# fig_save_path = os.path.join(fig_dir, "heterofl-ablation-10clients.pdf")
# fig_save_path = os.path.join(fig_dir, "fedavg-ablation-10clients.pdf")
# fig_save_path = os.path.join(fig_dir, "fedecover-ablation-100clients.pdf")
# fig_save_path = os.path.join(fig_dir, "fd-ablation-100clients.pdf")
# fig_save_path = os.path.join(fig_dir, "fedrolex-ablation-100clients.pdf")
# fig_save_path = os.path.join(fig_dir, "heterofl-ablation-100clients.pdf")
# fig_save_path = os.path.join(fig_dir, "fedavg-ablation-100clients.pdf")
# fig_save_path = os.path.join(fig_dir, "femnist-epochs3-num10.pdf")
# fig_save_path = os.path.join(fig_dir, "femnist-epochs3-num10.svg")
# fig_save_path = os.path.join(fig_dir, "femnist-epochs5-num10-gamma0.9-Tds100-Tdi5.pdf")
# fig_save_path = os.path.join(fig_dir, "gamma0.95-Tds100-Tdi10.svg")
# fig_save_path = os.path.join(fig_dir, "no-gsd.png")
# fig_save_path = os.path.join(fig_dir, "fedecover-iid.pdf")
# fig_save_path = os.path.join(fig_dir, "fd-iid.pdf")
# fig_save_path = os.path.join(fig_dir, "fedrolex-iid.pdf")
# fig_save_path = os.path.join(fig_dir, "heterofl-iid.pdf")

plt.figure()
plt.grid(True)

fedavg_mean_acc = None
reach_rounds = {}

for i, method in enumerate(methods):
    # Read the CSV file
    try:
        csv_path = os.path.join(
            csv_dir,
            f"{method}-{model}-{dataset}-{distribution}-{capacity}-{num_clients}.csv",
        )
        # csv_path = os.path.join(csv_dir, f"{method}-femnist-epochs3-num10.csv")
        # csv_path = os.path.join(csv_dir, f"{method}-no-gsd.csv")
        # csv_path = os.path.join(csv_dir, f"{method}-gamma0.95-Tds100-Tdi10.csv")
        # csv_path = os.path.join(
        #     csv_dir, f"{method}-femnist-epochs5-num10-gamma0.9-Tds100-Tdi5.csv"
        # )
        # csv_path = os.path.join(csv_dir, f"{method}-iid.csv")
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f'Warning: file "{csv_path}" not found.')
        continue

    # Extract the Round and Aggregated Test Acc columns
    rounds = df["Round"].iloc[:total_rounds]
    acc = df["Aggregated Test Acc"].iloc[: len(rounds)] * 100  # Convert to percentage

    # Plot the accuracy curve for the method
    label = method_labels.get(method, method)
    style = method_styles.get(method, {"linestyle": "-", "marker": "o", "markevery": 20, "linewidth": 2, "markersize": 6})
    
    plt.plot(rounds, acc, label=f"{label}", 
             color="black", 
             linewidth=style["linewidth"], 
             linestyle=style["linestyle"], 
             marker=style["marker"], 
             markevery=style["markevery"], 
             markersize=style["markersize"],
             markerfacecolor="white",
             markeredgecolor="black",
             markeredgewidth=1.5)

    # Calculate mean and standard deviation after avg_line_start_round
    acc_after_avg_line_start_round = acc[rounds > avg_line_start_round]
    mean_acc = acc_after_avg_line_start_round.mean()
    std_acc = acc_after_avg_line_start_round.std()
    print(f"{label} Mean: {mean_acc:.2f}%, Std: {std_acc:.2f}%")

    if method == "fedavg":
        fedavg_mean_acc = mean_acc

    try:
        first_reach_round = rounds[acc >= fedavg_mean_acc].iloc[0]
        reach_rounds[method] = first_reach_round
        if method == "fedavg":
            fedavg_reach_round = first_reach_round
    except IndexError:
        first_reach_round = None

for method in methods:
    label = method_labels.get(method, method)
    if reach_rounds.get(method, None) is not None:
        speedup = fedavg_reach_round / reach_rounds[method]
        print(
            f"{label} reaches FedAvg mean accuracy at round {reach_rounds[method]}, speedup: {speedup:.2f}"
        )
    else:
        print(f"{label} never reaches FedAvg mean accuracy")

plt.legend(fontsize=12, loc="lower right")
plt.xlabel("通信轮数", fontsize=20, fontfamily="WenQuanYi Zen Hei")
plt.ylabel("准确率 (%)", fontsize=20, fontfamily="WenQuanYi Zen Hei")
plt.tick_params(axis="both", which="major", labelsize=16)  # Increase tick size
# plt.ylim(40, 80)

plt.savefig(fig_save_path, bbox_inches="tight")
print(f"Figure saved at {fig_save_path}")
print("-" * 50)
