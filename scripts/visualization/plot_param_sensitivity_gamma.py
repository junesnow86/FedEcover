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

# Read data from csv files
accuracy_df = pd.read_csv(
    "results/param-sensitivity/different-gamma-Tds200-Tdi10-accuracy.csv"
)
speedup_df = pd.read_csv(
    "results/param-sensitivity/different-gamma-Tds200-Tdi10-speedup.csv"
)
baseline_df = pd.read_csv("results/param-sensitivity/baseline-accuracy.csv")

methods = accuracy_df["Method"].tolist()
gammas = accuracy_df.columns[1:].astype(float).tolist()

accuracy = {
    method: accuracy_df[accuracy_df["Method"] == method].values[0][1:].tolist()
    for method in methods
}
speedup = {
    method: speedup_df[speedup_df["Method"] == method].values[0][1:].tolist()
    for method in methods
}
baseline_accuracy = {
    method: baseline_df[baseline_df["Method"] == method].values[0][1]
    for method in methods
}

# Create a figure and set the x-axis label and y-axis label
fig, ax1 = plt.subplots()
ax1.set_xlabel("Gamma", fontsize=20)
ax1.set_ylabel("Accuracy (%)", fontsize=20)

# Create a second y-axis for the speedup
ax2 = ax1.twinx()
ax2.set_ylabel("Speedup", fontsize=20)

bar_width = 0.1
index = np.arange(len(gammas))

# Plot the accuracy line chart
for method in methods:
    ax1.plot(
        index + bar_width * (len(methods) - 1) / 2,
        accuracy[method],
        marker="o",
        label=f"{method_labels[method]}",
        color=method_colors[method],
    )

    # Add horizontal lines to indicate the baseline accuracy
    ax1.axhline(
        y=baseline_accuracy[method],
        linestyle="--",
        color=method_colors[method],
        # label=f"{method_labels[method]} baseline",
    )

ax1.tick_params(axis="y")
ax1.set_xticks(
    index + bar_width * (len(methods) - 1) / 2
)  # Set the x-axis ticks to the midpoints of the bars in the bar chart.
ax1.set_xticklabels(
    gammas, fontdict={"fontsize": 16}
)  # Set the x-axis tick labels to gamma values.
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
ax2.set_xticks(index + bar_width * (len(methods) - 1) / 2)
ax2.set_xticklabels(gammas)
ax2.set_yticklabels(ax2.get_yticks(), fontdict={"fontsize": 16})

# Create a single legend for both line and bar charts
# lines, labels = ax1.get_legend_handles_labels()
# bars, bar_labels = ax2.get_legend_handles_labels()
# ax2.legend(
#     lines + bars, labels + bar_labels, loc="center left", bbox_to_anchor=(1, 0.5)
# )
# ax2.legend(loc="upper left")
# ax2.legend(loc="upper center")
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.175), ncol=3, fontsize=12)
# plt.subplots_adjust(
#     top=0.95
# )  # Adjust the spacing between the chart and the boundaries.
plt.savefig(
    "figures/param-sensitivity/different-gamma-Tds200-Tdi10.pdf", bbox_inches="tight"
)
