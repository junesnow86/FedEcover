import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

sub_dir = "different-alpha"
fig_dir = "figures"
csv_dir = "results"
if sub_dir:
    csv_dir = os.path.join(csv_dir, sub_dir)
    fig_dir = os.path.join(fig_dir, sub_dir)
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

# Read data from csv files
# accuracy_df = pd.read_csv("alpha_effects_accuracy_10clients.csv")
# speedup_df = pd.read_csv("alpha_effects_speedup_10clients.csv")
accuracy_df = pd.read_csv(
    os.path.join(csv_dir, "different_alpha_accuracy_100clients.csv")
)
speedup_df = pd.read_csv(os.path.join(csv_dir, "different_alpha_speedup_100clients.csv"))

methods = accuracy_df["Method"].tolist()
alphas = accuracy_df.columns[1:].astype(float).tolist()

accuracy = {
    method: accuracy_df[accuracy_df["Method"] == method].values[0][1:].tolist()
    for method in methods
}
speedup = {
    method: speedup_df[speedup_df["Method"] == method].values[0][1:].tolist()
    for method in methods
}

# Create a figure and set the x-axis label and y-axis label
fig, ax1 = plt.subplots()
ax1.set_xlabel("Alpha", fontsize=20)
ax1.set_ylabel("Accuracy (%)", fontsize=20)

# Create a second y-axis for the speedup
ax2 = ax1.twinx()
ax2.set_ylabel("Speedup", fontsize=20)

bar_width = 0.1
index = np.arange(len(alphas))

# Plot the accuracy line chart
for method in methods:
    ax1.plot(
        index + bar_width * (len(methods) - 1) / 2,
        accuracy[method],
        marker="o",
        label=f"{method_labels[method]}",
        color=method_colors[method],
    )

ax1.tick_params(axis="y")
ax1.set_xticks(
    index + bar_width * (len(methods) - 1) / 2
)  # Set the x-axis ticks to the midpoints of the bars in the bar chart.
ax1.set_xticklabels(
    alphas, fontdict={"fontsize": 16}
)  # Set the x-axis tick labels to alpha values.
ax1.set_yticklabels(ax1.get_yticks(), fontdict={"fontsize": 16})

# Plot a bar chart of speedup.
bars = []
for i, method in enumerate(methods):
    bar = ax2.bar(
        index + i * bar_width,
        speedup[method],
        bar_width,
        label=f"{method_labels[method]}",
        alpha=0.7,
        color=method_colors[method],
    )
    bars.append(bar)

ax2.tick_params(axis="y")
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.175), ncol=3, fontsize=12)
ax2.set_xticks(index + bar_width * (len(methods) - 1) / 2)
ax2.set_xticklabels(alphas, fontdict={"fontsize": 16})
ax2.set_yticklabels(ax2.get_yticks(), fontdict={"fontsize": 16})

# plt.subplots_adjust(
#     top=0.95
# )  # Adjust the spacing between the chart and the boundaries.
plt.savefig(os.path.join(fig_dir, "different-alpha-100clients.pdf"), bbox_inches="tight")
