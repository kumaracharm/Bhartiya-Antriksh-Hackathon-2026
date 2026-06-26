import cv2
from preprocessing.preprocess import preprocess
from models.swinir import enhance
from models.pix2pix import colorize

def run_pipeline(image_path):
    print("Starting AI pipeline...")

    # Load image
    img = cv2.imread(image_path)

    if img is None:
        print("ERROR: Image not found at", image_path)
        return

    # Step 1: Preprocess
    img = preprocess(img)
    print("Preprocessing complete")

    # Step 2: Enhancement (SwinIR placeholder / future real model)
    img = enhance(img)
    print("SwinIR enhancement step complete")

    # Step 3: Colorization (Pix2Pix placeholder / future real model)
    img = colorize(img)
    print("Pix2Pix colorization step complete")

    # Step 4: Save output
    output_path = "outputs/final_output.jpg"
    cv2.imwrite(output_path, img)

    print("Pipeline completed. Output saved to", output_path)


if __name__ == "__main__":
    run_pipeline("data/raw/test.jpg")