import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 读取 CSV 文件
df = pd.read_csv("results/1010/fedrame_resnet_tiny-imagenet_alpha0.5_coverage_overlap.csv")

eta = (df["Coverage"] * 0.5 + df["Overlap"] * 0.5)

# 计算均值和方差
mean_coverage = df["Coverage"].mean()
std_coverage = df["Coverage"].std()
mean_overlap = df["Overlap"].mean()
std_overlap = df["Overlap"].std()
mean_eta = eta.mean()

# 绘制 Overlap 随 Round 变化的曲线
plt.figure(figsize=(10, 6))
plt.plot(df["Round"], df["Coverage"], marker="o", label="Coverage")
plt.plot(df["Round"], df["Overlap"], marker="o", label="Overlap")
plt.plot(df["Round"], eta, marker="o", label="Eta")


# 绘制均值和方差
plt.axhline(
    mean_coverage, linestyle="--", label=f"Mean: {mean_coverage:.4f}"
)
plt.axhline(
    mean_overlap, linestyle="--", label=f"Mean: {mean_overlap:.4f}"
)
plt.axhline(
    mean_eta, linestyle="--", label=f"Mean: {mean_eta:.4f}"
)
# plt.axhline(
#     mean_overlap + std_overlap,
#     color="g",
#     linestyle="--",
#     label=f"Mean + Std: {mean_overlap + std_overlap:.4f}",
# )
# plt.axhline(
#     mean_overlap - std_overlap,
#     color="g",
#     linestyle="--",
#     label=f"Mean - Std: {mean_overlap - std_overlap:.4f}",
# )

# 添加图例和标签
plt.title("Coverage and Overlap Ratio vs Round")
plt.xlabel("Round")
plt.ylabel("Ratio")
plt.legend()
plt.grid(True)

# 保存图像
plt.savefig("figures/1010/coverage_overlap_curve.png")

