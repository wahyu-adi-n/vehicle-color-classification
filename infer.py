from PIL import Image
from torchvision import transforms as T
from model.vcm_cnn import VehicleColorModel
from config import *
from utils.lib import *
import torch
import time

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
    image = Image.open(IMAGE_PATH).convert('RGB')
    label, time = infer(image)
    print(f"This image is predicted to {label}. Running on {time/1000:.4f}s")
