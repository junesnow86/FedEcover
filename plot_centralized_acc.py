import matplotlib.pyplot as plt
import pandas as pd

# 读取CSV文件
df = pd.read_csv("accuracy_records.csv")

# 提取数据
epochs = df["Epoch"]
train_accuracy = df["Train Accuracy"]
validation_accuracy = df["Validation Accuracy"]

# 计算最大值
max_train_accuracy = train_accuracy.max()
max_validation_accuracy = validation_accuracy.max()

# 画出曲线
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accuracy, label="Train Accuracy", marker="o", color="r")
plt.plot(
    epochs, validation_accuracy, label="Validation Accuracy", marker="x", color="b"
)

# 画出最大值的虚线
plt.axhline(
    y=max_train_accuracy,
    color="r",
    linestyle="--",
    label=f"Max Train Accuracy: {max_train_accuracy:.4f}",
)
plt.axhline(
    y=max_validation_accuracy,
    color="b",
    linestyle="--",
    label=f"Max Validation Accuracy: {max_validation_accuracy:.4f}",
)

# 添加标题和标签
plt.title("Train and Validation Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# 显示图像
plt.savefig("accuracy_plot.png")
