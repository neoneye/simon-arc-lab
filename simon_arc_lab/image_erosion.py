import numpy as np
from enum import Enum

class ImageErosionId(Enum):
    ALL8 = 'all8'
    NEAREST4 = 'nearest4'
    CORNER4 = 'corners4'


def image_erosion(image: np.array, erosion_id: ImageErosionId) -> np.array:
    """
    Erosion with mask.

    :param image: 2D numpy array with 0s and 1s.
    :param erosion_id: The type of erosion to apply.
    :return: 2D numpy array with 0s and 1s.
    """

    kernel = None
    if erosion_id == ImageErosionId.ALL8:
        # Create a 3x3 kernel with all ones
        kernel = np.ones((3, 3), dtype=np.uint8)
    elif erosion_id == ImageErosionId.NEAREST4:
        kernel = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]], dtype=np.uint8)
    elif erosion_id == ImageErosionId.CORNER4:
        kernel = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]], dtype=np.uint8)
    else:
        raise ValueError(f'Unknown erosion_id: {erosion_id}')
    
    # count number of 1s in the kernel
    number_of_ones_in_kernel = np.sum(kernel)

    height, width = image.shape

    # Convolve the image with the kernel
    output = np.zeros(image.shape, dtype=np.uint8)
    for y in range(height - 2):
        for x in range(width - 2):
            count = 0
            for dy in range(3):
                for dx in range(3):
                    if kernel[dy, dx] > 0 and image[y + dy, x + dx] > 0:
                        count += 1
            if count == number_of_ones_in_kernel:
                output[y + 1, x + 1] = 1

    return output
