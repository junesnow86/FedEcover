import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

# Read data from csv files
accuracy_df = pd.read_csv("alpha_effects_accuracy_10clients.csv")
speedup_df = pd.read_csv("alpha_effects_speedup_10clients.csv")

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
ax1.set_xlabel("Alpha", fontsize=14)
ax1.set_ylabel("Accuracy (%)", fontsize=14)

# Create a second y-axis for the speedup
ax2 = ax1.twinx()
ax2.set_ylabel("Speedup", fontsize=14)

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
ax1.set_xticklabels(alphas)  # Set the x-axis tick labels to alpha values.

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
ax2.legend(loc="upper left")
ax2.set_xticks(
    index + bar_width * (len(methods) - 1) / 2
)
ax2.set_xticklabels(alphas)

plt.subplots_adjust(top=0.95)  # Adjust the spacing between the chart and the boundaries.
plt.savefig("alpha_effects_comparison_10clients.png")
