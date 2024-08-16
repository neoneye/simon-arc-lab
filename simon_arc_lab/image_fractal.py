import numpy as np

def image_fractal_from_mask(mask: np.ndarray) -> np.ndarray:
    """
    Create a self-similar fractal mask from a mask image.

    Everywhere the input mask is 1, the entire input mask is copied to the output image.
    Everywhere the input mask is 0, the output image is 0.

    The output image size is the input size squared.

    :param mask: The input mask image.
    :return: The output fractal image. 
    """
    input_height, input_width = mask.shape
    output_width = input_width * input_width
    output_height = input_height * input_height
    output_image = np.zeros((output_height, output_width), dtype=np.uint8)
    for y in range(input_height):
        for x in range(input_width):
            if mask[y, x] > 0:
                for dy in range(input_height):
                    for dx in range(input_width):
                        output_image[y * input_height + dy, x * input_width + dx] = mask[dy, dx]

    return output_image
