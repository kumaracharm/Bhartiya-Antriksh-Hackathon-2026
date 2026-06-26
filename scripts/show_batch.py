import cv2
import os
import matplotlib.pyplot as plt
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
    batch_size=4,
    shuffle=True
)

batch=next(iter(loader))

for i in range(4):

    plt.figure()

    img=batch[i].permute(1,2,0)

    plt.imshow(img)

    plt.title(f"Image {i+1}")

    plt.axis("off")

    plt.savefig(f"outputs/image_{i+1}.png")

print("Images saved in outputs folder")

plt.show(block=True)