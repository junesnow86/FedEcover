import matplotlib.pyplot as plt
import pandas as pd

# ---------- Plotting Single Method Subset and Aggregated Performance ----------
df = pd.read_csv("results_0815_0/random_dropout_scale_small_models_square_200rounds.csv")

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

plt.plot(
    df["Round"],
    df["Aggregated"],
    label="Aggregated",
    color="black",
    linewidth=2,
)


plt.title(
    "Subset and Aggregated Performance Over Rounds (Random Dropout with Scale(square) Small Models)"
)
plt.xlabel("Round")
plt.ylabel("Performance")
plt.legend()
plt.grid(True)
plt.savefig("figures/0815/random_dropout_scale_small_models_square.png")