import cv2
import numpy as np

def preprocess(img):
    """
    Input: BGR image (numpy array from cv2)
    Output: normalized RGB image (float32)
    """

    if img is None:
        raise ValueError("Input image is None")

    # Resize to model standard
    img = cv2.resize(img, (256, 256))

    # Convert BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize [0,255] → [0,1]
    img = img.astype(np.float32) / 255.0

    return img