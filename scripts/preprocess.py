import cv2
import matplotlib.pyplot as plt

img = cv2.imread("data/raw/test.jpg")

print("Original:", img.shape)

resized = cv2.resize(img,(256,256))

print("Resized:", resized.shape)

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(resized,cv2.COLOR_BGR2RGB))
plt.title("Resized")
plt.axis("off")

plt.show()