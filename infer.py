from PIL import Image
from torchvision import transforms as T
from model.vcm import VehicleColorModel
from config import *
from utils.lib import *
import torch
import time
import os

model = VehicleColorModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))

transforms = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor()]
)


def infer(image):
    image = transforms(image)
    image = image.unsqueeze(0)
    t0 = time.time()
    pred = model.forward(image).argmax(dim=1)
    runtime = time.time() - t0
    class_label = decode_label(pred)
    return class_label, runtime


if __name__ == "__main__":
    start_time = time.time()

    for dirpath, dirnames, filenames in os.walk(IMAGE_PATH):
        count = 0
        other_class_predicted = []
        for file in filenames:
            image = dirpath + "/" + file
            image = Image.open(image).convert('RGB')
            label, runtime = infer(image)
            if str(label) == 'white':
                count += 1
            else:
                other_class_predicted.append(str(label))
            print(
                f"This image [{file}] is predicted to {label}. Running on {runtime/1000}s")
    print(f'Time inference all image is {time.time() - start_time:.4f}s')
    print(
        f'Num of image in {IMAGE_PATH} directory is {len(os.listdir(IMAGE_PATH))}')
    print(
        f'Num of image correct while inference is {count} out of {len(os.listdir(IMAGE_PATH))}')
    print(
        f'Accuracy for this class is {float(count/len(os.listdir(IMAGE_PATH))*100):.2f}%')
    print(f'Other class predicted is {set(other_class_predicted)}')
