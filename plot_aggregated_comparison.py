import matplotlib.pyplot as plt
import pandas as pd

# ---------- Plotting Aggregated Performance Comparison ----------
df1 = pd.read_csv("results_0726/vanilla_fedavg_unbalanced_classes.csv")
df2 = pd.read_csv("results_0726/heterofl_more_different_p_unbalanced_classes.csv")
df3 = pd.read_csv("results_0726/random_dropout_more_different_p_unbalanced_classes.csv")
df4 = pd.read_csv(
    "results_0726/position_heterofl_more_different_p_unbalanced_classes.csv"
)
df5 = pd.read_csv("results/random_dropout_more_small_models_unbalanced.csv")

# 提取数据
rounds_df1 = df1["Round"]
aggregated_df1 = df1["Aggregated"]
rounds_df2 = df2["Round"]
aggregated_df2 = df2["Pruned-global Aggregated"]
rounds_df3 = df3["Round"]
aggregated_df3 = df3["Aggregated"]
rounds_df4 = df4["Round"]
aggregated_df4 = df4["Aggregated"]
rounds_df5 = df5["Round"]
aggregated_df5 = df5["Aggregated"]

# 计算第50轮之后的平均值
avg_aggregated_df1 = aggregated_df1[rounds_df1 > 50].mean()
avg_aggregated_df2 = aggregated_df2[rounds_df2 > 50].mean()
avg_aggregated_df3 = aggregated_df3[rounds_df3 > 50].mean()
avg_aggregated_df4 = aggregated_df4[rounds_df4 > 50].mean()
avg_aggregated_df5 = aggregated_df5[rounds_df5 > 50].mean()

# 绘制图表
plt.figure(figsize=(10, 6))
color1 = "blue"
color2 = "green"
color3 = "orange"
color4 = "red"
color5 = "purple"
plt.plot(rounds_df1, aggregated_df1, label="Vanilla FedAvg", color=color1)
plt.plot(rounds_df2, aggregated_df2, label="HeteroFL", color=color2)
plt.plot(rounds_df3, aggregated_df3, label="Random Dropout (Ours)", color=color3)
plt.plot(rounds_df4, aggregated_df4, label="More-covered HeteroFL (Ours)", color=color4)
plt.plot(
    rounds_df5,
    aggregated_df5,
    label="Random Dropout with More Small Models (Ours)",
    color=color5,
)

# 绘制平均值线
plt.axhline(
    y=avg_aggregated_df1,
    color=color1,
    linestyle="--",
    label=f"Avg Vanilla FedAvg (after Round 50): {avg_aggregated_df1:.4f}",
)
plt.axhline(
    y=avg_aggregated_df2,
    color=color2,
    linestyle="--",
    label=f"Avg HeteroFL (after Round 50): {avg_aggregated_df2:.4f}",
)
plt.axhline(
    y=avg_aggregated_df3,
    color=color3,
    linestyle="--",
    label=f"Avg Random Dropout (after Round 50): {avg_aggregated_df3:.4f}",
)
plt.axhline(
    y=avg_aggregated_df4,
    color=color4,
    linestyle="--",
    label=f"Avg More-covered HeteroFL (after Round 50): {avg_aggregated_df4:.4f}",
)
plt.axhline(
    y=avg_aggregated_df5,
    color=color5,
    linestyle="--",
    label=f"Avg Random Dropout with More Small Models (after Round 50): {avg_aggregated_df5:.4f}",
)

# 添加图例和标签
# plt.title("Aggregated Performance Comparison on CIFAR-10")
plt.title("Aggregated Performance Comparison on CIFAR-10 (Unbalanced Classes)")
plt.xlabel("Round")
plt.ylabel("Aggregated Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("figures/0802/aggregated_performance_comparison_unbalanced.png")
# plt.savefig("figures/0726/aggregated_performance_comparison_unbalanced_classes.png")
