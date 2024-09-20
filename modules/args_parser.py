import argparse


def get_args(print_args=True):
    """Initialize the global parser configuration.

    Returns:
        args: Argparse namespace with all configs.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save the results",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=100,
        help="Number of clients to simulate",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn", "resnet", "shallow_resnet"],
        default="cnn",
        help="Model to use for training",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100"],
        default="cifar10",
        help="Dataset to use for training",
    )
    parser.add_argument(
        "--distribution",
        type=str,
        choices=["iid", "non-iid"],
        default="iid",
        help="Data distribution for clients",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Alpha value for Dirichlet distribution",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=200,
        help="Number of rounds to train",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs of local training",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for local training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for local training",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        choices=["sparse", "recovery"],
        default="sparse",
        help="Aggregation method to use",
    )
    parser.add_argument(
        "--debugging",
        type=bool,
        default=False,
        help="Debugging mode",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["fedavg", "heterofl", "fedrolex", "fedrd", "rdbagging", "legacy"],
        default="fedavg",
        help="Federated learning method",
    )
    parser.add_argument(
        "--select-ratio",
        type=float,
        default=0.1,
        help="Ratio of selected clients per round",
    )
    parser.add_argument(
        "--local-train-ratio",
        type=float,
        default=0.8,
        help="Ratio of local training data",
    )
    parser.add_argument(
        "--plot-data-distribution",
        type=bool,
        default=False,
        help="Plot data distribution",
    )
    parser.add_argument(
        "--rdbagging-strategy",
        type=str,
        default=None,
        help="RDBagging strategy",
    )

    args = parser.parse_args()

    if print_args:
        print(f"Random seed: {args.seed}")
        print(f"Method: {args.method}")
        print(f"Model type: {args.model}")
        print(f"Dataset: {args.dataset}")
        print(f"Data distribution: {args.distribution}")
        if args.distribution == "non-iid":
            print(f"Alpha: {args.alpha}")
        print(f"Aggregation method: {args.aggregation}")
        print(f"Number of clients: {args.num_clients}")
        print(f"Select ratio: {args.select_ratio}")
        print(f"Local train ratio: {args.local_train_ratio}")
        print(f"Number of rounds: {args.rounds}")
        print(f"Number of local epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.lr}")
        print(f"Results save directory: {args.save_dir}")
        if args.save_dir is None:
            print("Results will not be saved.")

    return args
