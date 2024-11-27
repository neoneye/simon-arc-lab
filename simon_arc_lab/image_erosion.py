import numpy as np
from .pixel_connectivity import PixelConnectivity
from scipy.ndimage import binary_erosion

def image_erosion(image: np.array, pixel_connectivity: PixelConnectivity) -> np.array:
    """
    Erosion with mask.

    :param image: 2D numpy array with 0s and 1s.
    :param pixel_connectivity: The type of erosion to apply.
    :return: 2D numpy array with 0s and 1s.
    """

    kernel = None
    if pixel_connectivity == PixelConnectivity.ALL8:
        # Create a 3x3 kernel with all ones
        kernel = np.ones((3, 3), dtype=np.uint8)
    elif pixel_connectivity == PixelConnectivity.NEAREST4:
        kernel = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]], dtype=np.uint8)
    elif pixel_connectivity == PixelConnectivity.CORNER4:
        kernel = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]], dtype=np.uint8)
    elif pixel_connectivity == PixelConnectivity.LR2:
        kernel = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0]], dtype=np.uint8)
    elif pixel_connectivity == PixelConnectivity.TB2:
        kernel = np.array([
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0]], dtype=np.uint8)
    elif pixel_connectivity == PixelConnectivity.TLBR2:
        kernel = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]], dtype=np.uint8)
    elif pixel_connectivity == PixelConnectivity.TRBL2:
        kernel = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]], dtype=np.uint8)
    else:
        raise ValueError(f'Unknown pixel connectivity: {pixel_connectivity}')
    
    output = binary_erosion(image, structure=kernel).astype(np.uint8)
    return output
