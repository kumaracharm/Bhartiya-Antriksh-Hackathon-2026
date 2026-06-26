import cv2
import torch
from torchvision import transforms

img = cv2.imread("data/raw/test.jpg")

print("Original shape:", img.shape)

transform = transforms.ToTensor()

tensor_img = transform(img)

print("Tensor shape:", tensor_img.shape)

print("Datatype:", tensor_img.dtype)
