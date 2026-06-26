import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os


class ImageDataset(Dataset):

    def __init__(self, folder):
        self.folder = folder
        self.images = os.listdir(folder)
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        path = os.path.join(
            self.folder,
            self.images[index]
        )

        img = cv2.imread(path)

        img = self.transform(img)

        return img


dataset = ImageDataset("data/raw")

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True
)

for img in loader:

    print("Batch shape:", img.shape)

    break