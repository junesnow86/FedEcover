import matplotlib.pyplot as plt
import numpy as np


def draw(method):
    file_path = f"results/param-selection-count/{method}.npy"
    count = np.load(file_path)
    mean_count = np.mean(count)
    diff = count - mean_count

    plt.figure()
    plt.bar(range(len(diff)), diff, linewidth=5)
    plt.xlabel("Neuron index", fontsize=16)
    plt.ylabel("Selection Count (Diff from Mean)", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if method == "fedecover":
        plt.ylim(-20, 20)

    save_path = f"figures/neuron-selection-count/{method}.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved to {save_path}")


for method in ["heterofl", "fedrolex", "fd", "fedecover"]:
    draw(method)
