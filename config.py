import torch
import torchvision.transforms as T

IMAGE_PATH = "./inference/test/9.jpg"
FOLDER_PATH = './inference/test'
MODEL_PATH = './output/models/VCM_CNN_5/last.pt'
DEVICE = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

TRANSFORMS = transforms = T.Compose([
    T.Resize(size=(256, 256)),
    T.CenterCrop(size=(224, 224)),
    T.ToTensor()]
)

LABELS = ['black', 'blue', 'gray',  'red', 'white']
NUM_CLASS = len(LABELS)

SAVE_PLOT_PATH = "output/save_plot/"
SAVE_CM_PATH = "output/save_cm/"
