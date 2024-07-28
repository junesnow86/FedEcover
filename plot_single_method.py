import matplotlib.pyplot as plt
import pandas as pd

# ---------- Plotting Single Method Subset and Aggregated Performance ----------
df = pd.read_csv("results_0726/random_dropout_more_different_p_unbalanced_classes.csv")

# Plotting
plt.figure(figsize=(10, 8))

# Loop through each subset column
for i in range(1, 11):
    plt.plot(df["Round"], df[f"Subset {i}"], label=f"Subset {i}")

# Plot the aggregated column
# plt.plot(
#     df["Round"],
#     df["Pruned-global Aggregated"],
#     label="Pruned-global Aggregated",
#     color="black",
#     linewidth=2,
#     linestyle="--",
# )

# plt.plot(
#     df["Round"],
#     df["Whole Aggregated"],
#     label="Whole Aggregated",
#     color="blue",
#     linewidth=2,
#     linestyle="--",
# )

plt.plot(
    df["Round"],
    df["Aggregated"],
    label="Aggregated",
    color="black",
    linewidth=2,
    linestyle="--",
)


plt.title(
    "Subset and Aggregated Performance Over Rounds (Random Dropout with More Different p - Unbalanced Classes)"
)
plt.xlabel("Round")
plt.ylabel("Performance")
plt.legend()
plt.grid(True)
plt.savefig("figures/0726/random_dropout_more_different_p_unbalanced.png")