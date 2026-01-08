import tifffile as tf
import numpy as np
import os

# Point this to your local folder containing mask000.tif
path = "/home/osalamon/OPENsalam/WP3/DeepFuse/gold_detection_truth/Fluo-N2DH-GOWT1/SEG/01_GT_markerim/"
img = tf.imread(os.path.join(path, "man_seg001.tif"))

print(f"Success! Image shape: {img.shape}, Type: {img.dtype}")
# Expected output: Success! Image shape: (1024, 1024), Type: uint16 (or similar)