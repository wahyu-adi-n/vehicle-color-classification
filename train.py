import argparse
import torch
import os
from pathlib import Path
from data import dataset
from utils.engine import train_model
from model.vcm_cnn import VehicleColorModel


def main(args):
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device('cpu')
    os.makedirs("output", exist_ok=True)
    wd = Path(os.getcwd()) / "output"

    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    epochs = args.epochs
    lr = args.lr

    experiment_name = "VCMCNN"

    train_loader, test_loader = dataset.get_loaders(batch_size=batch_size,
                                                    test_batch_size=test_batch_size,
                                                    dataset_path="data/color/")

    print(
        f"================================ Training {experiment_name} ================================")
    params = {
        "batch_size": batch_size,
        "test_batch_size": test_batch_size,
    }
    model = VehicleColorModel()
    # print(model)
    # x = torch.randn(48, 3, 224, 224)
    # out = model(x)
    # print(out)
    train_model(model, train_loader, test_loader,
                f"{wd}/{experiment_name}", device, experiment_name,
                lr=lr, epochs=epochs, params_dict=params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vehicle Color Classification")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="Batch size for training data (default: 32)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=32,
        metavar="N",
        help="Batch size for validation and testing data (default: 32)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="Number of epochs for training (default: 10)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        metavar="LR",
        help="Learning rate for training (default: 1e-4)"
    )

    args = parser.parse_args()
    main(args)
