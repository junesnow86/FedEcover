import numpy as np


def create_non_iid_data(dataset, num_clients, alpha, seed=18):
    """Create non-IID data distribution for federated learning, using Dirichlet distribution.
    
    Args:
        dataset (torch.utils.data.Dataset): The dataset to split.
        num_clients (int): The number of clients.
        alpha (float): The concentration parameter of the Dirichlet distribution.

    Returns:
        Tuple[Dict[int, List[int]], Dict[int, Dict[str, Union[int, Dict[int, int]]]]]: 
            A tuple containing a dictionary of client indices and a dictionary of client statistics.
    """
    data_indices = np.arange(len(dataset))
    labels = np.array(dataset.targets)

    client_dict = {
        i: [] for i in range(num_clients)
    }  # the value is a list of indices that assigned to the client

    num_classes = len(np.unique(labels))
    np.random.seed(seed)

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
