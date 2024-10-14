import argparse


def get_args(print_args=True):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        type=str,
        choices=[
            "fedavg",
            "heterofl",
            "fedrolex",
            "fedrd",
            "fedrame",
        ],
        default="fedavg",
        help="Federated learning method",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn", "resnet"],
        default="cnn",
        help="Model type",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100", "tiny-imagenet"],
        default="cifar10",
        help="Dataset",
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default="iid",
        help="Data distribution for clients",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=100,
        help="Number of clients to simulate",
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
        default=32,
        help="Batch size for local training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for local training",
    )
    parser.add_argument(
        "--eta_g",
        type=float,
        default=1.0,
        help="Eta_g for aggregation",
    )
    parser.add_argument(
        "--dynamic-eta_g",
        type=str,
        default="False",
        help="Dynamic Eta_g for aggregation",
    )
    parser.add_argument(
        "--plot-data-distribution",
        type=str,
        default="False",
        help="Plot data distribution",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--norm-type",
        type=str,
        choices=["sbn", "ln"],
        default="ln",
        help="Normalization type for ResNet model",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of workers for DataLoader",
    )

    args = parser.parse_args()

    if args.plot_data_distribution == "True":
        args.plot_data_distribution = True
    else:
        args.plot_data_distribution = False

    if args.dynamic_eta_g == "True":
        args.dynamic_eta_g = True
    else:
        args.dynamic_eta_g = False

    if print_args:
        print(f"Method: {args.method}")
        print(f"Model: {args.model}")
        print(f"Dataset: {args.dataset}")
        print(f"Data distribution: {args.distribution}")
        if args.distribution != "iid":
            try:
                args.distribution = float(args.distribution.split("alpha")[1])
            except ValueError:
                raise ValueError("Alpha value must be a float")
        print(f"Dynamic Global learning rate: {args.dynamic_eta_g}")
        print(f"Global learning rate: {args.eta_g}")
        print(f"Number of clients: {args.num_clients}")
        print(f"Client selection ratio: {args.select_ratio}")
        print(f"Local training data ratio: {args.local_train_ratio}")
        print(f"Number of rounds: {args.rounds}")
        print(f"Number of local epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.lr}")
        print(f"Random seed: {args.seed}")
        print(f"Normalization type: {args.norm_type}")
        print(f"Number of workers: {args.num_workers}")

    return args
