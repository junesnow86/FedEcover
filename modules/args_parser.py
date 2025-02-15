import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # Basic settings
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=[
            "fedavg",
            "heterofl",
            "fedrolex",
            "fd",
            "fedecover",
        ],
        default="fedavg",
        help="Federated learning method",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn", "femnistcnn", "resnet"],
        default="cnn",
        help="Model type",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100", "tiny-imagenet", "celeba", "femnist"],
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
        "--client-select-mode",
        type=str,
        choices=["num", "ratio"],
        default="ratio",
        help="Client selection mode",
    )
    parser.add_argument(
        "--client-select-num",
        type=int,
        default=10,
        help="Number of selected clients per round",
    )
    parser.add_argument(
        "--client-select-ratio",
        type=float,
        default=0.1,
        help="Ratio of selected clients per round",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=300,
        help="Number of rounds to train",
    )
    parser.add_argument(
        "--plot-data-distribution",
        type=str,
        default="False",
        help="Plot data distribution",
    )
    parser.add_argument(
        "--norm-type",
        type=str,
        choices=["sbn", "ln"],
        default="sbn",
        help="Normalization type for ResNet model",
    )
    parser.add_argument(
        "--client-capacity-distribution",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help="Which client capacity distribution group to use",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers for DataLoader",
    )

    # Local training settings
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
        "--data-augmentation",
        type=str,
        default="False",
        help="Data augmentation",
    )

    # Server settings
    parser.add_argument(
        "--eta_g",
        type=float,
        default=1.0,
        help="Global aggregation lr",
    )
    parser.add_argument(
        "--global-lr-decay",
        type=str,
        default="False",
        help="Whether to use global learning rate decay",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Decay coefficient for GSD",
    )
    parser.add_argument(
        "--Tds",
        type=int,
        default=200,
        help="Stop round for GSD",
    )
    parser.add_argument(
        "--Tdi",
        type=int,
        default=10,
        help="Gamma decay update interval rounds",
    )

    args = parser.parse_args()

    args.plot_data_distribution = (
        True if args.plot_data_distribution == "True" else False
    )
    args.global_lr_decay = True if args.global_lr_decay == "True" else False
    args.data_augmentation = True if args.data_augmentation == "True" else False

    if args.distribution != "iid":
        try:
            args.distribution = float(args.distribution.split("alpha")[1])
        except ValueError:
            raise ValueError("Alpha value must be a float")

    return args
