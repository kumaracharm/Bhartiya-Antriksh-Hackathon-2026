from preprocessing.preprocess import preprocess
from inference.swinir_model import run_swinir
from inference.pix2pix_model import run_pix2pix
import cv2

print("Starting AI pipeline...")

input_path = "data/raw/test.jpg"

# Step 1: Preprocess
img = cv2.imread(input_path)
processed = preprocess(img)
print("Preprocessing complete")

# Step 2: Enhancement
enhanced = run_swinir(processed)
print("SwinIR enhancement step complete")

# Step 3: Colorization
final_output = run_pix2pix(enhanced)
print("Pix2Pix colorization step complete")

# Step 4: Save output
cv2.imwrite("outputs/final_output.jpg", final_output)

print("Pipeline completed.")
print("Saved at: outputs/final_output.jpg")