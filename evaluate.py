from model.vcm import models
from data import dataset
from torchvision import transforms as T
from utils.lib import *
from sklearn.metrics import classification_report
import torch
import argparse


BASE_DIR = "output/models/"
DEVICE = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_model(args):
    model = models(args)
    model.load_state_dict(torch.load(
        BASE_DIR+'VCM_CNN_5/weights_epoch_50.pt', map_location=DEVICE))
    return model


def predict(model, image):
    model.eval()
    with torch.no_grad():
        pred = model.forward(image).argmax(dim=1)
    return pred


def eval_data_preparation(model, args):
    _, test_loader = dataset.data_loaders(batch_size=32,
                                          test_batch_size=32,
                                          dataset_path="data/color/",
                                          num_workers=2, test_size=args)

    test_predict, true_label = [], []

    for image, label in test_loader:
        pred = predict(model, image)
        test_predict.append(pred[0].item())
        true_label.append(label[0].item())

    return test_predict, true_label


def main(args):
    model = load_model(args.model)
    test_predict, true_label = eval_data_preparation(model, args.test_size)
    print(classification_report(true_label, test_predict))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Vehicle Color Evaluation")

    parser.add_argument(
        "--model",
        type=str,
        default='vcmcnn',
        metavar="M",
        help="Model that is used for training."
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.5,
        metavar="TS",
        help="Test Size for training (default: 1e-4)"
    )

    args = parser.parse_args()
    main(args)
