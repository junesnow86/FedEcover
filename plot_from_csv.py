import matplotlib.pyplot as plt
import pandas as pd

# ---------- Plotting Single Method Subset and Aggregated Performance ----------
# # Load the CSV file
# # df = pd.read_csv("statistics/0720/heterofl_unbalanced_classes_100rounds_0720.csv")
# # df = pd.read_csv(
# #     "statistics/0720/position_heterofl_unbalanced_classes_100rounds_0720.csv"
# # )
# # df = pd.read_csv("results/position_heterofl.csv")
# df = pd.read_csv("results/position_heterofl_unbalanced_classes.csv")

# # Plotting
# plt.figure(figsize=(10, 8))

# # Loop through each subset column
# for i in range(1, 11):
#     plt.plot(df["Round"], df[f"Subset {i}"], label=f"Subset {i}")

# # Plot the aggregated column
# plt.plot(
#     df["Round"],
#     df["Aggregated"],
#     label="Aggregated",
#     color="black",
#     linewidth=2,
#     linestyle="--",
# )

# # plt.title(
# #     "Subset and Aggregated Performance Over Rounds (HeteroFL - Unbalanced Classes)"
# # )
# plt.title(
#     "Subset and Aggregated Performance Over Rounds (More-covered HeteroFL - Unbalanced Classes)"
# )
# # plt.title(
# #     "Subset and Aggregated Performance Over Rounds (Vanilla FedAvg - Unbalanced Classes)"
# # )
# # plt.title(
# #     "Subset and Aggregated Performance Over Rounds (Hetero Random Dropout - Unbalanced Classes)"
# # )
# plt.xlabel("Round")
# plt.ylabel("Performance")
# plt.legend()
# plt.grid(True)
# # plt.savefig("figures/0720/heterofl_unbalanced_classes_100rounds_0720.png")
# # plt.savefig("figures/0720/position_heterofl_unbalanced_classes_100rounds_0720.png")
# # plt.savefig("figures/0721/position_overall.png")
# plt.savefig("figures/0721/position_overall_unbalanced_classes.png")


# ---------- Plotting Aggregated Performance Comparison ----------
# df1 = pd.read_csv(
#     "statistics/0720/position_heterofl_unbalanced_classes_100rounds_0720.csv"
# )
# df2 = pd.read_csv("statistics/0720/heterofl_unbalanced_classes_100rounds_0720.csv")

# df1 = pd.read_csv("results/vanilla_fedavg.csv")
# df2 = pd.read_csv("results/heterofl.csv")
# df3 = pd.read_csv("results/hetero_random_dropout.csv")
# df4 = pd.read_csv("results/position_heterofl.csv")
df1 = pd.read_csv("results/vanilla_fedavg_unbalanced_classes.csv")
df2 = pd.read_csv("results/heterofl_unbalanced_classes.csv")
df3 = pd.read_csv("results/hetero_random_dropout_unbalanced_classes.csv")
df4 = pd.read_csv("results/position_heterofl_unbalanced_classes.csv")

# 提取数据
rounds_df1 = df1["Round"]
aggregated_df1 = df1["Aggregated"]
rounds_df2 = df2["Round"]
aggregated_df2 = df2["Aggregated"]
rounds_df3 = df3["Round"]
aggregated_df3 = df3["Aggregated"]
rounds_df4 = df4["Round"]
aggregated_df4 = df4["Aggregated"]

# 计算第20轮之后的平均值
avg_aggregated_df1_after_20 = aggregated_df1[rounds_df1 > 20].mean()
avg_aggregated_df2_after_20 = aggregated_df2[rounds_df2 > 20].mean()
avg_aggregated_df3_after_20 = aggregated_df3[rounds_df3 > 20].mean()
avg_aggregated_df4_after_20 = aggregated_df4[rounds_df4 > 20].mean()

# 绘制图表
plt.figure(figsize=(10, 6))
color1 = "blue"
color2 = "green"
color3 = "orange"
color4 = "red"
plt.plot(rounds_df1, aggregated_df1, label="Vanilla FedAvg", color=color1)
plt.plot(rounds_df2, aggregated_df2, label="HeteroFL", color=color2)
plt.plot(rounds_df3, aggregated_df3, label="Hetero Random Dropout", color=color3)
plt.plot(rounds_df4, aggregated_df4, label="More-covered HeteroFL", color=color4)

# 绘制平均值线
plt.axhline(
    y=avg_aggregated_df1_after_20,
    color=color1,
    linestyle="--",
    label="Avg Vanilla FedAvg (after Round 20)",
)
plt.axhline(
    y=avg_aggregated_df2_after_20,
    color=color2,
    linestyle="--",
    label="Avg HeteroFL (after Round 20)",
)
plt.axhline(
    y=avg_aggregated_df3_after_20,
    color=color3,
    linestyle="--",
    label="Avg Hetero Random Dropout (after Round 20)",
)
plt.axhline(
    y=avg_aggregated_df4_after_20,
    color=color4,
    linestyle="--",
    label="Avg More-covered HeteroFL (after Round 20)",
)

# 添加图例和标签
# plt.title("Aggregated Performance Comparison on CIFAR-10")
plt.title("Aggregated Performance Comparison on CIFAR-10 (Unbalanced Classes)")
plt.xlabel("Round")
plt.ylabel("Aggregated Accuracy")
plt.legend()
plt.grid(True)
# plt.savefig("figures/0721/aggregated_performance_comparison.png")
plt.savefig("figures/0721/aggregated_performance_comparison_unbalanced_classes.png")


# ---------- Plotting Class-wise Performance Comparison ----------
# # 读取CSV文件
# # data = pd.read_csv("statistics/0720/heterofl_aggregated_class_wise_acc_results.csv")
# # data = pd.read_csv(
# #     "statistics/0720/position_heterofl_aggregated_class_wise_acc_results.csv"
# # )
# data = pd.read_csv("statistics/0721/position_aggregated_class_wise_acc_results.csv")

# # 绘制图表
# plt.figure(figsize=(10, 8))
# for column in data.columns[1:]:  # 跳过第一列（Round）
#     plt.plot(data["Round"], data[column], label=column)

# # plt.title("Aggregated Class-wise Accuracy over Rounds (Vanilla FedAvg)")
# # plt.title("Aggregated Class-wise Accuracy over Rounds (HeteroFL)")
# plt.title("Aggregated Class-wise Accuracy over Rounds (More-covered HeteroFL)")
# # plt.title("Aggregated Class-wise Accuracy over Rounds (Hetero Random Dropout)")
# plt.xlabel("Round")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.grid(True)
# # plt.savefig("figures/0720/heterofl_aggregated_class_wise_acc.png")
# plt.savefig("figures/0721/position_aggregated_class_wise_acc.png")
