NORMALIZATION_STATS = {
    "cifar10": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010]},
    "cifar100": {"mean": [0.5071, 0.4867, 0.4408], "std": [0.2675, 0.2565, 0.2761]},
    "imagenet": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "tiny-imagenet": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "celeba": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
}

OPTIONAL_CLIENT_CAPACITY_DISTRIBUTIONS = [
    [
        # Expected values for the distribution is 0.3
        (1.0, 0.05),
        (0.75, 0.1),
        (0.5, 0.15),
        (0.25, 0.2),
        (0.1, 0.5),
    ],
    [
        # Uniform distribution, expected values for the distribution is 0.52
        (1.0, 0.2),
        (0.75, 0.2),
        (0.5, 0.2),
        (0.25, 0.2),
        (0.1, 0.2),
    ],
    [
        # Distribution for small amount of clients
        (1.0, 0.1),
        (0.5, 0.4),
        (0.1, 0.5),
    ],
    [
        # Uniform distribution for small amount of clients
        (1.0, 0.33),
        (0.5, 0.33),
        (0.1, 0.34),
    ],
    [
        (0.5, 0.5),
        (0.1, 0.5),
    ],
]
