import cv2
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class SatelliteDataset(Dataset):

    def __init__(self, folder):

        self.folder = folder

        self.images = os.listdir(folder)

        self.transform = transforms.Compose([

            transforms.ToPILImage(),

            transforms.Resize((256,256)),

            transforms.ToTensor()

        ])

    def __len__(self):

        return len(self.images)

    def __getitem__(self,index):

        path=os.path.join(
            self.folder,
            self.images[index]
        )

        img=cv2.imread(path)

        img=self.transform(img)

        return img


dataset=SatelliteDataset("data/raw")

loader=DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

for batch in loader:

    print("Batch:",batch.shape)