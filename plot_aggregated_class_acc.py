import matplotlib.pyplot as plt
import pandas as pd

# ---------- Plotting Class-wise Performance Comparison ----------
data = pd.read_csv(
    "statistics/0726/random_dropout_aggregated_class_wise_acc_results.csv"
)

# 绘制图表
plt.figure(figsize=(10, 8))
for column in data.columns[1:]:  # 跳过第一列（Round）
    plt.plot(data["Round"], data[column], label=column)

# plt.title("Aggregated Class-wise Accuracy over Rounds (Vanilla FedAvg)")
# plt.title("Aggregated Class-wise Accuracy over Rounds (HeteroFL)")
# plt.title("Aggregated Class-wise Accuracy over Rounds (More-covered HeteroFL)")
plt.title("Aggregated Class-wise Accuracy over Rounds (Random Dropout)")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("figures/0726/random_dropout_aggregated_class_wise_acc.png")
