import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 读取 CSV 文件
df = pd.read_csv("results/1010/fedrame_resnet_tiny-imagenet_alpha0.5_sparsity.csv")

# 计算均值和方差
mean_sparsity = df["Sparsity"].mean()
std_sparsity = df["Sparsity"].std()

# 绘制 Sparsity 随 Round 变化的曲线
plt.figure(figsize=(10, 6))
plt.plot(df["Round"], df["Sparsity"], marker="o", label="Sparsity")

# 绘制均值和方差
plt.axhline(
    mean_sparsity, color="r", linestyle="--", label=f"Mean: {mean_sparsity:.4f}"
)
plt.axhline(
    mean_sparsity + std_sparsity,
    color="g",
    linestyle="--",
    label=f"Mean + Std: {mean_sparsity + std_sparsity:.4f}",
)
plt.axhline(
    mean_sparsity - std_sparsity,
    color="g",
    linestyle="--",
    label=f"Mean - Std: {mean_sparsity - std_sparsity:.4f}",
)

# 添加图例和标签
plt.title("Sparsity vs Round")
plt.xlabel("Round")
plt.ylabel("Sparsity")
plt.legend()
plt.grid(True)

# 保存图像
plt.savefig("figures/1010/sparsity_curve.png")


# --------------------------------------------------
# 对 Sparsity 数据进行反平方根变换
transformed_sparsity = np.sqrt(df["Sparsity"])

# 将变换后的数据归一化到 (0, 1) 范围内
# normalized_sparsity = (transformed_sparsity - transformed_sparsity.min()) / (transformed_sparsity.max() - transformed_sparsity.min())

# 计算变换后数据的均值和方差
mean_sparsity = transformed_sparsity.mean()
std_sparsity = transformed_sparsity.std()

# 绘制变换后 Sparsity 随 Round 变化的曲线
plt.figure(figsize=(10, 6))
plt.plot(df["Round"], transformed_sparsity, marker="o", label="Transformed Sparsity")

# 绘制均值和方差
plt.axhline(
    mean_sparsity, color="r", linestyle="--", label=f"Mean: {mean_sparsity:.4f}"
)
plt.axhline(
    mean_sparsity + std_sparsity,
    color="g",
    linestyle="--",
    label=f"Mean + Std: {mean_sparsity + std_sparsity:.4f}",
)
plt.axhline(
    mean_sparsity - std_sparsity,
    color="g",
    linestyle="--",
    label=f"Mean - Std: {mean_sparsity - std_sparsity:.4f}",
)

# 添加图例和标签
plt.title("Transformed Sparsity vs Round")
plt.xlabel("Round")
plt.ylabel("Transformed Sparsity")
plt.legend()
plt.grid(True)

plt.savefig("figures/1010/transformed_sparsity_curve.png")
