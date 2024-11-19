"""
Test script to test the huang_thresholding function.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from huang.thresholding import huang_thresholding


if __name__ == "__main__":

    IMAGES_PATH = Path("test_images")
    input_image = "." / IMAGES_PATH / "lymp.png"
    output_image = "." / IMAGES_PATH / "lymp_binary.png"

    img = plt.imread(input_image)
    img = (255 * (img - img.min()) / (img.max() - img.min())).astype(np.uint8)

    thresh, _ = huang_thresholding(img)
    thresholded_img = (img > thresh) * 1.0

    plt.imsave(output_image, thresholded_img, cmap="gray")
