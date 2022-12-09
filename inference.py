from PIL import Image
from torchvision import transforms as T
from model.vcm import models
from config import *
from utils.lib import *
import torch
import time
import argparse


def load_model(args):
    model = models(args)
    model.load_state_dict(torch.load(
        MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()
    return model


def inference(args, image):
    image = TRANSFORMS(image)
    model = load_model(args)
    model.eval()
    outputs = model(image.unsqueeze(0))
    predict = torch.softmax(outputs[0], dim=0)
    conf_score = torch.max(predict).item()
    class_label = decode_label(predict.argmax())
    return class_label, conf_score*100


def main(args):
    img = Image.open(args.image_path).convert('RGB')

    if args.show_img:
        img.show()

    class_labels, conf_score = inference(args.model, img)
    print(
        f"This image is predicted to {class_labels}. Conf score is {(conf_score*100):.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Vehicle Color Inference")
    parser.add_argument(
        "--model",
        type=str,
        default="vcmcnn"
    )

    parser.add_argument(
        "--image-path",
        type=str,
        default=IMAGE_PATH
    )

    parser.add_argument(
        '--show-img',
        type=bool,
        default=False
    )

    args = parser.parse_args()
    main(args)
