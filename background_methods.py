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

# ------------------------
# corner sampling background estimation
# ------------------------

# WŁASNE: define bilinear interpolation for corner background estimation
def bilinear_interpolation(height, width, v_00, v_10, v_01, v_11):
    B = np.zeros((height, width), dtype=np.float64)

    for i in range(height):
        for j in range(width):
            u = j / (width - 1)
            v = i / (height - 1)

            B[i, j] = (1 - u)*(1 - v)*v_00 + u*(1 - v)*v_10 + (1 - u)*v*v_01 + u*v*v_11

    return B

# WŁASNE: define corner background estimation using bilinear interpolation of the four corners
def corner_background(image: np.ndarray, corner_size: int, sigma: float):
    height, width = image.shape

    # GENAI: Claude – fix of corner-to-v_uv mapping (slicing order in numpy)
    # numpy indexing is [row, column] = [v, u]:  rows  :s -> top,   -s: -> bottom
    #                                            cols  :s -> left,  -s: -> right
    # v_00: u=0, v=0 -> left-top corner
    v_00 = np.mean(image[ :corner_size, :corner_size])
    # v_10: u=1, v=0 -> right-top corner
    v_10 = np.mean(image[ :corner_size, -corner_size:])
    # v_01: u=0, v=1 -> left-bottom corner
    v_01 = np.mean(image[-corner_size: , :corner_size])
    # v_11: u=1, v=1 -> right-bottom corner
    v_11 = np.mean(image[-corner_size: , -corner_size:])

    B = bilinear_interpolation(height, width, v_00, v_10, v_01, v_11)
    R_1 = image - B

    radius = int(np.ceil(3 * sigma))
    size = 2 * radius + 1
    R_2 = gaussian_filter(image.astype(np.float64), sigma=sigma, size=size) - B

    return B, R_1, R_2





