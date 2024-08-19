import matplotlib.pyplot as plt
import pandas as pd

# ---------- Plotting Aggregated Performance Comparison ----------
df1 = pd.read_csv("results_0726/vanilla_fedavg.csv")
df2 = pd.read_csv("results_0726/heterofl_more_different_p.csv")
df3 = pd.read_csv("results_0819/rd_1x_test_acc.csv")

# 截取前100轮
df3 = df3.iloc[:100]

# 提取数据
rounds_df1 = df1["Round"]
aggregated_df1 = df1["Aggregated"]
rounds_df2 = df2["Round"]
aggregated_df2 = df2["Pruned-global Aggregated"]
# aggregated_df2 = df2["Aggregated"]
rounds_df3 = df3["Round"]
aggregated_df3 = df3["Aggregated"]

# 计算某一轮之后的平均值
avg_aggregated_df1 = aggregated_df1[rounds_df1 > 60].mean()
avg_aggregated_df2 = aggregated_df2[rounds_df2 > 60].mean()
avg_aggregated_df3 = aggregated_df3[rounds_df3 > 60].mean()

# 计算最大值
max_aggregated_df1 = aggregated_df1.max()
max_aggregated_df2 = aggregated_df2.max()
max_aggregated_df3 = aggregated_df3.max()

# 绘制变化曲线
plt.figure(figsize=(10, 8))
color1 = "blue"
color2 = "green"
color3 = "red"
plt.plot(rounds_df1, aggregated_df1, label="Vanilla FedAvg", color=color1)
plt.plot(rounds_df2, aggregated_df2, label="HeteroFL", color=color2)
plt.plot(rounds_df3, aggregated_df3, label="Random Dropout (Ours)", color=color3)

# 绘制平均值线
plt.axhline(
    y=avg_aggregated_df1,
    color=color1,
    linestyle="--",
    label=f"Avg Vanilla FedAvg (after Round 60): {avg_aggregated_df1:.4f}",
)
plt.axhline(
    y=avg_aggregated_df2,
    color=color2,
    linestyle="--",
    label=f"Avg HeteroFL (after Round 60): {avg_aggregated_df2:.4f}",
)
plt.axhline(
    y=avg_aggregated_df3,
    color=color3,
    linestyle="--",
    label=f"Avg Random Dropout (after Round 60): {avg_aggregated_df3:.4f}",
)

# 绘制最大值线
plt.axhline(
    y=max_aggregated_df1,
    color=color1,
    linestyle="-.",
    label=f"Max Vanilla FedAvg: {max_aggregated_df1:.4f}",
)
plt.axhline(
    y=max_aggregated_df2,
    color=color2,
    linestyle="-.",
    label=f"Max HeteroFL: {max_aggregated_df2:.4f}",
)
plt.axhline(
    y=max_aggregated_df3,
    color=color3,
    linestyle="-.",
    label=f"Max Random Dropout: {max_aggregated_df3:.4f}",
)

# 添加图例和标签
plt.title("Aggregated Performance Comparison on CIFAR-10")
# plt.title("Aggregated Performance Comparison on CIFAR-10 (Unbalanced Classes)")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("figures/0819/aggregated_performance_comparison.png")
