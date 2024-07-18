import matplotlib.pyplot as plt
import pandas as pd

# 读取数据
data = pd.read_csv(
    "statistics/hetero_2_party_aggregation_test_epochs_5_p_0.5_0.5_no_scaling.csv"
)

# 提取数据列
rounds = data["Round"]
pruned_accuracy_1 = data["Pruned Model 1 Accuracy"]
pruned_accuracy_2 = data["Pruned Model 2 Accuracy"]
aggregated_accuracy = data["Aggregated Model Accuracy"]

# 创建图形
plt.figure(figsize=(14, 8))

# 绘制 Pruned Model 1 Accuracy 曲线
plt.plot(
    rounds,
    pruned_accuracy_1,
    label="Pruned Model 1 Accuracy (p=0.5)",
    marker="o",
    linestyle="-",
)

# 绘制 Pruned Model 2 Accuracy 曲线
plt.plot(
    rounds,
    pruned_accuracy_2,
    label="Pruned Model 2 Accuracy (p=0.5)",
    marker="x",
    linestyle="--",
)

# 绘制 Aggregated Model Accuracy 曲线
plt.plot(
    rounds,
    aggregated_accuracy,
    label="Aggregated Model Accuracy",
    marker="s",
    linestyle="-.",
)

# 添加标题和标签
plt.title("Model Accuracy vs Rounds")
plt.xlabel("Rounds")
plt.ylabel("Accuracy")

# 添加图例
plt.legend()

# 添加网格
plt.grid(True)

# 保存图形到文件
plt.savefig("figures/hetero_2_party_aggregation_test_epochs_5_p_0.5_0.5_no_scaling.png")

# 关闭图形
plt.close()
