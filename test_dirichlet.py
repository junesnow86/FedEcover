import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# CIFAR-10
# train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# CIFAR-100
train_dataset = datasets.CIFAR100(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.CIFAR100(
    root="./data", train=False, download=True, transform=transform
)


def create_non_iid_data(dataset, num_clients, alpha):
    data_indices = np.arange(len(dataset))
    labels = np.array(dataset.targets)

    client_dict = {
        i: [] for i in range(num_clients)
    }  # the value is a list of indices that assigned to the client

    num_classes = len(np.unique(labels))
    np.random.seed(18)

    for k in range(num_classes):
        class_indices = data_indices[labels == k]
        np.random.shuffle(class_indices)

        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]

        class_indices_split = np.split(class_indices, proportions)

        for i in range(num_clients):
            client_dict[i] += class_indices_split[i].tolist()

    # Calculate the number of samples and class distribution for each client
    client_stats = {}
    for i in range(num_clients):
        client_indices = client_dict[i]
        client_labels = labels[client_indices]
        class_counts = {
            cls: int(np.sum(client_labels == cls)) for cls in range(num_classes)
        }
        client_stats[i] = {
            "total_samples": len(client_indices),
            "class_distribution": class_counts,
        }

    return client_dict, client_stats


num_clients = 100
alpha = 0.5
client_data_indices, client_stats = create_non_iid_data(
    train_dataset, num_clients, alpha
)

# Print the total number of samples and class distribution for each client
for client_id, stats in client_stats.items():
    print(f"Client {client_id}:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Class distribution: {stats['class_distribution']}")




alpha_values = [0.1, 0.5, 1.0, 10.0, 100.0]
num_clients = 3
num_classes = 4

fig, axes = plt.subplots(1, len(alpha_values), figsize=(20, 5))
colors = ['b', 'g', 'r', 'y']

for i, alpha in enumerate(alpha_values):
    np.random.seed(42)
    proportions = np.random.dirichlet(np.repeat(alpha, num_clients), size=num_classes)

    for j in range(num_classes):
        axes[i].bar(np.arange(num_clients) + j*num_clients, proportions[j], color=colors[j], alpha=0.7, label=f'Class {j}' if i == 0 else "")
    
    axes[i].set_title(f"alpha = {alpha}")
    axes[i].set_xlabel('Client-Class')
    axes[i].set_ylabel('Proportion')
    # axes[i].legend()

plt.savefig("dirichlet_distribution.png")
