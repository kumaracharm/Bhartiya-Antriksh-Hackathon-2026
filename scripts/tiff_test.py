import numpy as np
import tifffile
import matplotlib.pyplot as plt

# Create a fake satellite image
fake_satellite = np.random.randint(
    0,
    255,
    (256,256),
    dtype=np.uint8
)

# Save as TIFF
tifffile.imwrite(
    "data/raw/sample_satellite.tif",
    fake_satellite
)

print("TIFF image created")


img=tifffile.imread(
    "data/raw/sample_satellite.tif"
)

print("Shape:",img.shape)

print("Datatype:",img.dtype)

plt.imshow(img,cmap="gray")
plt.title("Satellite TIFF")
plt.axis("off")
plt.show()