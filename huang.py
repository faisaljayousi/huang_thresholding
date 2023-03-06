import numpy as np
from loops import (_mu_loop,
                   _entropy_loop)


def huang_thresholding(image: np.ndarray) -> int:
    r""" Computes and returns threshold and corresponding entropy using Huang's fuzzy thresholding method. [1]

        [1] Huang L.-K. and Wang M.-J.J. (1995) "Image Thresholding by Minimizing the Measures of Fuzziness"
    """

    if np.issubdtype(image.dtype, np.floating):
        raise ValueError("Float images are not supported by huang_thresholding. "
                         "Convert the image to an unsigned integer type.")

    if np.issubdtype(image.dtype, np.signedinteger) and np.any(image < 0):
        raise ValueError("Negative-valued images are not supported.")

    data, _ = np.histogram(image, bins=range(256))

    first_bin, last_bin = np.min(image), np.max(image)
    length = last_bin - first_bin

    mu0 = np.ascontiguousarray([*range(256)], dtype=np.float64)
    mu1 = np.ascontiguousarray([*range(256)], dtype=np.float64)

    _mu_loop(data, mu0, mu1, first_bin, last_bin)

    # Compute entropy
    threshold, min_entropy = _entropy_loop(
        data, 1 / length, mu0, mu1, np.float('inf'), -1)

    return threshold, min_entropy
