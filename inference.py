from PIL import Image
from torchvision import transforms as T
from model.vcm import VehicleColorModel
from config import *
from utils.lib import *
import torch
import time
import os
import argparse
import matplotlib.pyplot as plt


def load_model():
    model = VehicleColorModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    return model


def inference(image):
    image = TRANSFORMS(image)
    t0 = time.time()
    model = load_model()
    pred = model.forward(image.unsqueeze(0)).argmax(dim=1)
    runtime = time.time() - t0
    class_label = decode_label(pred)
    return class_label, runtime


def main(args):
    img = Image.open(args.image_path).convert('RGB')

    if args.show_img:
        img.show()

    label, runtime = inference(img)
    print(
        f"This image is predicted to {label}. Time inference is {runtime/1000:.4f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Vehicle Color Inference")
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

    # for dirpath, dirnames, filenames in os.walk(IMAGE_PATH):
    #     # count = 0
    #     # other_class_predicted = []
    #     times = 0
    #     for file in filenames:
    #         image = dirpath + "/" + file
    #         image = Image.open(image).convert('RGB')
    #         label, runtime = inference(image)
    #         # if str(label) == 'white':
    #         #     count += 1
    #         # else:
    #         #     other_class_predicted.append(str(label))
    #         times += runtime
    #         print(
    #             f"This image [{file}] is predicted to {label}. Running on {runtime/1000}s")
    # print(f'Time inference all image is {times/1000:.4f}s')
    # print(f'Time inference all image is {time.time() - start_time:.4f}s')
    # print(
    #     f'Num of image in {IMAGE_PATH} directory is {len(os.listdir(IMAGE_PATH))}')
    # print(
    #     f'Num of image correct while inference is {count} out of {len(os.listdir(IMAGE_PATH))}')
    # print(
    #     f'Accuracy for this class is {float(count/len(os.listdir(IMAGE_PATH))*100):.2f}%')
    # print(f'Other class predicted is {set(other_class_predicted)}')
