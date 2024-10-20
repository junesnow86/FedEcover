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
        "--client-select-ratio",
        type=float,
        default=0.1,
        help="Ratio of selected clients per round",
    )
    parser.add_argument(
        "--local-train-ratio",
        type=float,
        default=1.0,
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
        default=10,
        help="Number of epochs of local training",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for local training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for local training",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for local training",
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
        default="sbn",
        help="Normalization type for ResNet model",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of workers for DataLoader",
    )
    parser.add_argument(
        "--client-capacity-distribution",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help="Which client capacity distribution group to use",
    )
    parser.add_argument(
        "--param-delta-norm",
        type=str,
        default="mean",
        choices=["mean", "uniform"],
        help="Parameter delta norm type",
    )
    parser.add_argument(
        "--global-lr-decay",
        type=str,
        default="False",
        help="Global learning rate decay",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Gamma for learning rate decay",
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

    if args.global_lr_decay == "True":
        args.global_lr_decay = True
    else:
        args.global_lr_decay = False

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
        print(f"Global step size: {args.eta_g}")
        print(f"Dynamic Global step size: {args.dynamic_eta_g}")
        print(f"Number of clients: {args.num_clients}")
        print(f"Client selection ratio: {args.client_select_ratio}")
        print(f"Local training data ratio: {args.local_train_ratio}")
        print(f"Number of rounds: {args.rounds}")
        print(f"Number of local epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.lr}")
        print(f"Weight decay: {args.weight_decay}")
        print(f"Random seed: {args.seed}")
        print(f"Normalization type: {args.norm_type}")
        print(f"Number of workers: {args.num_workers}")
        print(f"Client capacity distribution: {args.client_capacity_distribution}")
        print(f"Parameter delta norm type: {args.param_delta_norm}")
        print(f"Global learning rate decay: {args.global_lr_decay}")
        print(f"Gamma: {args.gamma}")

    return args
