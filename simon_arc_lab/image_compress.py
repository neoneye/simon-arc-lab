import numpy as np

def image_compress_x(image: np.array) -> np.array:
    """
    Eliminate adjacent duplicate columns
    """
    if image.shape[1] == 0:
        return image
    # Initialize a list with the first column
    compressed = [image[:, 0]]
    # Iterate over the array columns starting from the second column
    for col in range(1, image.shape[1]):
        # Compare with the last column in the compressed list
        if not np.array_equal(image[:, col], compressed[-1]):
            compressed.append(image[:, col])
    return np.array(compressed).T

def image_compress_y(image: np.array) -> np.array:
    """
    Eliminate adjacent duplicate rows
    """
    if len(image) == 0:
        return image
    # Initialize a list with the first row
    compressed = [image[0]]
    # Iterate over the array starting from the second row
    for row in image[1:]:
        # Compare with the last row in the compressed list
        if not np.array_equal(row, compressed[-1]):
            compressed.append(row)
    return np.array(compressed)

def image_compress_xy(image: np.array) -> np.array:
    """
    Eliminate adjacent duplicate rows+columns
    """
    return image_compress_y(image_compress_x(image))
