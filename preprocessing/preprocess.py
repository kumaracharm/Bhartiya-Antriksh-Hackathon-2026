import cv2

def preprocess(image_path):

    img=cv2.imread(image_path)

    img=cv2.resize(img,(256,256))

    print("Preprocessing complete")

    return img