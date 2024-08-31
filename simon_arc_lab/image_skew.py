import numpy as np

def image_skew_right(image: np.array, padding_color: int) -> np.array:
    """
    Displace each row to the right by the row index.
    """
    height, width = image.shape
    skewed_image = np.full((height, height + width - 1), padding_color, dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            skewed_image[y, y+x] = image[y, x]
    return skewed_image

def image_unskew_right(image: np.array) -> np.array:
    """
    Displace each row to the left by the row index. Remove the padding.
    """
    height, width = image.shape
    width_unskewed = width - height + 1
    unskewed_image = np.zeros((height, width_unskewed), dtype=np.uint8)
    for y in range(height):
        for x in range(width_unskewed):
            unskewed_image[y, x] = image[y, y+x]
    return unskewed_image

def image_skew_down(image: np.array, padding_color: int) -> np.array:
    """
    Displace each column down by the column index.
    """
    height, width = image.shape
    skewed_image = np.full((height + width - 1, width), padding_color, dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            skewed_image[y+x, x] = image[y, x]
    return skewed_image

def image_unskew_down(image: np.array) -> np.array:
    """
    Displace each column up by the column index. Remove the padding.
    """
    height, width = image.shape
    height_unskewed = height - width + 1
    unskewed_image = np.zeros((height_unskewed, width), dtype=np.uint8)
    for y in range(height_unskewed):
        for x in range(width):
            unskewed_image[y, x] = image[y+x, x]
    return unskewed_image
