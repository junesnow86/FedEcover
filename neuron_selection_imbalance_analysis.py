import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def draw_hist(file, bins=30, method=None):
    _, ax = plt.subplots()
    data = np.load(file)
    # data = data - np.mean(data)

    if method == "fedecover":
        bins = 10

    sns.histplot(
        data,
        bins=bins,
        kde=True,
        ax=ax,
        stat="percent",
    )
    ax.axvline(np.mean(data), color="r", linestyle="--", label="Mean")
    ax.axvline(np.median(data), color="g", linestyle="--", label="Median")
    ax.legend(fontsize=16)

    ax.set_xlabel("Neuron selection frequency", fontsize=20)
    ax.set_ylabel("Percentage of neurons", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=16)

    if method == "fedecover":
        x_ticks = np.arange(int(np.min(data) - 1), int(np.max(data) + 1) + 1, 1)
        ax.set_xticks(x_ticks)


for method in ["heterofl", "fedrolex", "fd", "fedecover"]:
    draw_hist(f"results/param-selection-count/{method}.npy", method=method)
    plt.tight_layout()
    plt.savefig(f"figures/neuron-selection-imbalance-analysis/{method}.pdf")
