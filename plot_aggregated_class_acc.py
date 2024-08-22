import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv("results_0819/aggregated_class_acc_base.csv")

# 假设CSV文件有以下列：Round, Class1, Class2, ..., ClassN
# 提取轮数
rounds = df["Round"]

# 提取类别列（假设从第二列开始）
classes = df.columns[1:]

# 绘制各个类别随轮数变化的曲线
plt.figure(figsize=(10, 8))
for class_name in classes:
    simplified_class_name = class_name.replace("_", " ").replace("Acc", "").strip()
    plt.plot(rounds, df[class_name], label=simplified_class_name)

# 添加图例
plt.legend()

# 添加标题和标签
plt.title("Class Accuracy Over Rounds (Base Unbalanced)")
plt.xlabel("Round")
plt.ylabel("Accuracy")

# 显示图形
plt.savefig("figures/0820/aggregated_class_acc_base_unbalanced.png")
