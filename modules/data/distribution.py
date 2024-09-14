import numpy as np


def create_non_iid_data(dataset, num_clients, alpha, seed=42):
    """Create non-IID data distribution for federated learning, using Dirichlet distribution.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to split.
        num_clients (int): The number of clients.
        alpha (float): The concentration parameter of the Dirichlet distribution.

    Returns:
        client_indices_list (list): A list of lists containing the indices of samples for each client.
        client_stats (list): A list of dictionaries containing the number of samples and class distribution for each client.
    """
    np.random.seed(seed)
    all_indices = np.arange(len(dataset))
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))
    client_indices_list = [[] for _ in range(num_clients)]

    for k in range(num_classes):
        class_indices = all_indices[labels == k]
        np.random.shuffle(class_indices)

        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]

        class_indices_split = np.split(class_indices, proportions)

        for i in range(num_clients):
            client_indices_list[i] += class_indices_split[i].tolist()

    # Calculate the number of samples and class distribution for each client
    client_stats = []
    for i in range(num_clients):
        client_indices = client_indices_list[i]
        client_labels = labels[client_indices]
        class_counts = {c: int(np.sum(client_labels == c)) for c in range(num_classes)}
        client_stats.append(
            {
                "total_samples": len(client_indices),
                "class_distribution": class_counts,
            }
        )

    return client_indices_list, client_stats
