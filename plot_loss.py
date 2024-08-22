import matplotlib.pyplot as plt
import pandas as pd

# Read the CSV files
df_train_loss = pd.read_csv("results_0821/rd_base_unbalanced_train_loss.csv")
df_val_loss = pd.read_csv("results_0821/rd_base_unbalanced_test_loss.csv")

# Create a figure and a single subplot
fig, ax = plt.subplots(figsize=(10, 8))

# Define colors for different subsets
colors = plt.cm.tab10(range(10))

# Plot each subset for train loss
for i in range(1, 11):
    ax.plot(
        df_train_loss["Round"],
        df_train_loss[f"Subset {i}"],
        label=f"Subset {i}",
        color=colors[i - 1],
        linestyle="-"
    )

# Plot each subset for validation loss
for i in range(1, 11):
    ax.plot(
        df_val_loss["Round"],
        df_val_loss[f"Subset {i}"],
        # label=f"Val Subset {i}",
        color=colors[i - 1],
        linestyle="--"
    )

# Plot the aggregated columns
ax.plot(
    df_val_loss["Round"],
    df_val_loss["Aggregated"],
    label="Aggregated Val Loss",
    color="black",
    marker="o",
)

# Customize the plot
ax.set_xlabel("Round")
ax.set_ylabel("Loss")
fig.suptitle("Subset and Aggregated Loss Over Rounds (Base - Unbalanced)", y=0.95)

# Combine legends from both axes
lines, labels = ax.get_legend_handles_labels()

# Add custom legend entries for line styles
custom_lines = [
    plt.Line2D([0], [0], linestyle='-', color='black'),
    plt.Line2D([0], [0], linestyle='--', color="black"),
]
custom_labels = ['Train Loss', 'Validation Loss']

ax.legend(lines + custom_lines, labels + custom_labels, loc="upper left")

plt.grid(True)
plt.savefig("figures/0821/rd_base_unbalanced_loss.png")