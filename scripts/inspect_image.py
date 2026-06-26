import cv2
import matplotlib.pyplot as plt

img = cv2.imread("data/raw/test.jpg")

print("Shape:", img.shape)

print("Min pixel:", img.min())
print("Max pixel:", img.max())

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.title("Loaded Image")
plt.axis("off")

plt.show()