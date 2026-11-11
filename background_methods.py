import numpy as np
from filters import gaussian_filter

# I(x,y) – input image in greyscale,
# B(x,y) – estimated background map,
# R(x,y) – image after background correction,
# Gσ(I) – image obtained by Gaussian filtering with parameter σ,
# SE – structural element used in morphological operations.

# ------------------------
# Gaussian blur background estimation 
# ------------------------

# WŁASNE: using Gaussian blur filter to estimate B and R
def blur_background(image: np.ndarray, sigma: float, size: int):
    B = gaussian_filter(image.astype(np.float64), sigma=sigma, size=size)
    # here B(x,y) = Gσ(I)(x,y)
    R = image - B

    return B, R

# ------------------------
# White Top-hat background estimation
# ------------------------

# WŁASNE: define the structuring element (SE) for morphological operations 
#           two options: square and disk
def make_structuring_element(shape: str, size: int):
    if shape == "Square":
        return np.ones((size, size), dtype=bool)
    elif shape == "Disk":
        radius = size // 2
        se = np.zeros((size, size), dtype=bool)
        for i in range(size):
            for j in range(size):
                if (i - radius) ** 2 + (j - radius) ** 2 <= radius ** 2:
                    se[i, j] = 1
        return se
    else:
        raise ValueError("Unsupported structuring element shape. Use 'Square' or 'Disk'.")

# WŁASNE: define erosion based on previous task adaptive thresholding logic (same windowing logic)
def erosion(image: np.ndarray, se: np.ndarray):
    window_radius = se.shape[0] // 2
    padded_image = np.pad(image, window_radius, mode='edge')
    eroded = np.zeros(image.shape)
    height, width = image.shape

    for x in range(height):
        for y in range(width):
            window = padded_image[x:x + se.shape[0], y:y + se.shape[1]]
            eroded[x, y] = np.min(window[se])

    return eroded

# WŁASNE: define dilation based on previous task adaptive thresholding logic (same windowing logic)
def dilation(image: np.ndarray, se: np.ndarray):
    window_radius = se.shape[0] // 2
    padded_image = np.pad(image, window_radius, mode='edge')
    dilated = np.zeros(image.shape)
    height, width = image.shape
    
    for x in range(height):
        for y in range(width):
            window = padded_image[x:x + se.shape[0], y:y + se.shape[1]]
            dilated[x, y] = np.max(window[se])

    return dilated

# WŁASNE: opening defined = erosion followed by dilation
def opening(image: np.ndarray, se: np.ndarray):
    eroded = erosion(image, se)
    opened = dilation(eroded, se)

    return opened

# WŁASNE: define tophat background estimation using morphological opening
def tophat_background(image: np.ndarray, se_shape: str, se_size: int):
    se = make_structuring_element(se_shape, se_size)
    B = opening(image, se)
    R = image - B

    return B, R


def corner_background(image: np.ndarray, corner_size: int, sigma: float):
    """Method 3: background from 4 corner samples + bilinear interpolation.

    corner_size: side of the square measuring area in each corner
    sigma:       Gaussian blur for the second variant

    Returns (B, R1, R2):  R1 = I - B,  R2 = G_sigma(I) - B
    """
    raise NotImplementedError("corner_background")
