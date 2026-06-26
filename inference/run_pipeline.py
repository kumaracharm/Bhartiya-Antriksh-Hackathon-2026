from preprocessing.preprocess import preprocess
from models.swinir import enhance
from models.pix2pix import colorize
import cv2

print("Starting AI pipeline...")

image_path = "data/raw/test.jpg"

processed = preprocess(image_path)
enhanced = enhance(processed)
result = colorize(enhanced)

# SAVE OUTPUT (important upgrade)
cv2.imwrite("outputs/final_output.jpg", result)

print("Pipeline completed. Output saved to outputs/final_output.jpg")