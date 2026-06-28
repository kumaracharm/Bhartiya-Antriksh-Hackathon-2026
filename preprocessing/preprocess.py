import cv2

def preprocess(img):
    if img is None:
        raise ValueError("Empty image input")

    img = cv2.resize(img, (256, 256))
    return img