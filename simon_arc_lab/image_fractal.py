import numpy as np

def image_fractal_from_mask(mask: np.ndarray) -> np.ndarray:
    """
    Create a self-similar fractal mask from a mask image.

    Everywhere the input mask is 1, the entire input mask is copied to the output image.
    Everywhere the input mask is 0, the output image is 0.

    The output image size is the input size squared.

    :param mask: The input mask.
    :return: The output fractal image. 
    """
    empty_color = 0
    return image_fractal_from_mask_and_image(mask, mask, empty_color)

def image_fractal_from_mask_and_image(mask: np.ndarray, image: np.ndarray, empty_color: int) -> np.ndarray:
    """
    Create a self-similar fractal mask from a mask image.

    Everywhere the input mask is 1, the entire input image is copied to the output image.
    Everywhere the input mask is 0, the output image is set to the empty_color.

    The output image size is the input mask x input image.

    :param mask: The input mask image.
    :param image: The input image to be inserted.
    :param empty_color: The background color for areas where the mask is 0.
    :return: The output fractal image. 
    """
    input_mask_height, input_mask_width = mask.shape
    input_image_height, input_image_width = mask.shape
    output_width = input_mask_width * input_image_width
    output_height = input_mask_height * input_image_height
    output_image = np.full((output_height, output_width), empty_color, dtype=np.uint8)
    for y in range(input_mask_height):
        for x in range(input_mask_width):
            if mask[y, x] > 0:
                for dy in range(input_image_height):
                    for dx in range(input_image_width):
                        output_image[y * input_image_height + dy, x * input_image_width + dx] = image[dy, dx]

    return output_image
