r"""
Test script to test the huang_thresholding function.
"""

import numpy as np
import matplotlib.pyplot as plt
from huang import huang_thresholding

# -----------------------------------------------------------------------------------

img = plt.imread(r'test_images/lymp.png')
img = (255 * (img - img.min()) / (img.max() - img.min())).astype(np.uint8)

thresh, _ = huang_thresholding(img)
thresholded_img = (img > thresh) * 1.

plt.imsave(r'./test_images/lymp_binary.png', thresholded_img, cmap='gray')
