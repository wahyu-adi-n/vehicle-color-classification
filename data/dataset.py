from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from utils.lib import *
from config import *
import glob
import torchvision.transforms as T

class VehicleColorDataset(Dataset):
    def __init__(self, image_list, class_list, transforms=None):
        self.transform = transforms
        self.image_list = image_list
        self.class_list = class_list
        self.data_len = len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.class_list[index]

    def __len__(self):
        return self.data_len


def data_loaders(batch_size: int,
                 test_batch_size: int,
                 dataset_path: str,
                 num_workers: int,
                 test_size: float):
    image_list = glob.glob(dataset_path + '**/*')
    class_list = [encode_label_from_path(item) for item in image_list]
    x_train, x_test, y_train, y_test = train_test_split(
        image_list, class_list, test_size=test_size, shuffle=True, random_state=42)
    print(len(x_train))
    print(len(x_test))

    train_dataset = VehicleColorDataset(x_train, y_train, TRANSFORMS)
    test_dataset = VehicleColorDataset(x_test, y_test, TRANSFORMS)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers)

    return train_loader, test_loader
