import numpy as np
from .pixel_connectivity import PixelConnectivity

def image_dilation(image: np.array, pixel_connectivity: PixelConnectivity) -> np.array:
    """
    Dilation with mask.

    :param image: 2D numpy array with 0s and 1s.
    :param pixel_connectivity: The type of dilation to apply.
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
    
    # Pad the image with a 1px border that is 0
    image_with_padding = np.zeros((image.shape[0] + 2, image.shape[1] + 2), dtype=np.uint8)
    image_with_padding[1:-1, 1:-1] = image  # Insert original image into the padded array

    height, width = image_with_padding.shape

    # Convolve the image with the kernel
    output_with_padding = np.zeros(image_with_padding.shape, dtype=np.uint8)
    for y in range(height - 2):
        for x in range(width - 2):
            for dy in range(3):
                for dx in range(3):
                    if kernel[dy, dx] > 0 and image_with_padding[y + dy, x + dx] > 0:
                        output_with_padding[y + 1, x + 1] = 1
                        break
                if output_with_padding[y + 1, x + 1] == 1:
                    break

    # remove the padding
    output = output_with_padding[1:-1, 1:-1]

    return output
