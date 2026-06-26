import cv2

print("Program started")

img = cv2.imread("data/raw/test.jpg")

print("Image object:", img)

if img is None:
    print("ERROR: Image not found")
else:
    print("Shape:", img.shape)
    print("Datatype:", img.dtype)