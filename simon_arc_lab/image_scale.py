from typing import Tuple
import numpy as np

def image_scale(unscaled_image: np.array, x_up_down: str, x_scale: int, y_up_down: str, y_scale: int) -> Tuple[np.array, np.array]:
    """
    Scale an image by a factor of x_scale in the x direction and y_scale in the y direction.

    :param unscaled_image: The image to scale.
    :param x_up_down: 'up' to scale up, 'down' to scale down.
    :param x_scale: The factor to scale by in the x direction.
    :param y_up_down: 'up' to scale up, 'down' to scale down.
    :param y_scale: The factor to scale by in the y direction.
    :return: A tuple of the input image and the output image.
    """
    # Input image
    input_image = unscaled_image.copy()
    if x_up_down == 'down':
        input_image = np.kron(input_image, np.ones((1, x_scale))).astype(np.uint8)
    if y_up_down == 'down':
        input_image = np.kron(input_image, np.ones((y_scale, 1))).astype(np.uint8)

    # Output image
    output_image = unscaled_image.copy()
    if x_up_down == 'up':
        output_image = np.kron(output_image, np.ones((1, x_scale))).astype(np.uint8)
    if y_up_down == 'up':
        output_image = np.kron(output_image, np.ones((y_scale, 1))).astype(np.uint8)
    return (input_image, output_image)

def image_scale_uniform(unscaled_image: np.array, up_down: str, scale: int) -> Tuple[np.array, np.array]:
    return image_scale(unscaled_image, up_down, scale, up_down, scale)
