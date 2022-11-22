from pathlib import Path
from data import dataset
from utils.engine import train_model
from model.vcm import models
from torch.optim import *
from config import *
import argparse
import torch
import os


def main(args):
    os.makedirs("output", exist_ok=True)
    wd = Path(os.getcwd()) / "output"

    # init model
    model = models(args.model)

    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    test_size = args.test_size
    epochs = args.epochs
    momentum = args.momentum
    lr = args.lr
    num_workers = args.num_workers
    experiment_name = "VCM"

    train_loader, test_loader = dataset.data_loaders(batch_size=batch_size,
                                                     test_batch_size=test_batch_size,
                                                     dataset_path="data/color/",
                                                     num_workers=num_workers, test_size=test_size)

    print("Training process is started!")

    print(model)

    train_model(model, train_loader, test_loader,
                f"{wd}/{experiment_name}", params={
                    "experiment_name": experiment_name,
                    "batch_size": batch_size,
                    "test_batch_size": test_batch_size,
                    "epochs": epochs,
                    "learning_rate": lr,
                    "momentum": momentum,
                    "num_workers": num_workers,
                    "device": DEVICE,
                    "model": args.model
                })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vehicle Color Classification")

    parser.add_argument(
        "--model",
        type=str,
        default='vcmcnn',
        metavar="M",
        help="Model that is used for training."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="BS",
        help="Batch size for training data (default: 32)"
    )

    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=32,
        metavar="TBS",
        help="Batch size for validation and testing data (default: 32)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="E",
        help="Number of epochs for training (default: 10)"
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        metavar="NW",
        help="Number of workers for training (default: 2)"
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="MM",
        help="Momentum for training (default: 0.9)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        metavar="LR",
        help="Learning rate for training (default: 1e-4)"
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        metavar="TS",
        help="Test Size for training (default: 1e-4)"
    )

    args = parser.parse_args()
    main(args)
