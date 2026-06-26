import cv2
from torchvision import transforms

img = cv2.imread("data/raw/test.jpg")

print("Original shape:", img.shape)

transform = transforms.Compose([
    
    transforms.ToPILImage(),

    transforms.Resize((256,256)),

    transforms.ToTensor()

])

processed = transform(img)

print("Processed shape:", processed.shape)

print("Min value:", processed.min())

print("Max value:", processed.max())