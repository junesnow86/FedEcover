import argparse


def get_args(print_args=True):
    """Initialize the global parser configuration.

    Returns:
        args: Argparse namespace with all configs.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save the results",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn", "resnet"],
        default="cnn",
        help="Model to use for training",
    )
    parser.add_argument(
        "--lr-decay",
        type=bool,
        default=False,
        help="Whether to use learning rate decay",
    )
    parser.add_argument(
        "--round",
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

    args = parser.parse_args()

    if print_args:
        print(f"Model type: {args.model}")
        print(f"Number of rounds: {args.round}")
        print(f"Number of local epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.lr}")
        print(f"Learning rate decay: {args.lr_decay}")
        print(f"Save directory: {args.save_dir}")
        if args.save_dir is None:
            print("Results will not be saved.")

    return args
