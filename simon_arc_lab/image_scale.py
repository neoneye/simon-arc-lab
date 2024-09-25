from typing import Tuple
import random
import numpy as np
from .image_util import *

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

def _image_scale_up_variable_y(unscaled_image: np.array, repeat_list: list[int]) -> np.array:
    """
    Scale an image by varying the number of times each rows is repeated.

    If the repeat count is zero, the row is removed.

    :param unscaled_image: The image to scale.
    :param repeat_list: The number of times a row is repeated.
    :return: The scaled image.
    """

    height, width = unscaled_image.shape
    if height != len(repeat_list):
        raise ValueError("The length of repeat_list should be equal to the width of the image.")
    
    scaled_image = np.zeros((sum(repeat_list), width), dtype=np.uint8)
    y = 0
    for i in range(height):
        for _ in range(repeat_list[i]):
            scaled_image[y, :] = unscaled_image[i, :]
            y += 1
    return scaled_image

def _image_scale_up_variable_x(unscaled_image: np.array, repeat_list: list[int]) -> np.array:
    """
    Scale an image by varying the number of times each column is repeated.

    If the repeat count is zero, the row is column.

    :param unscaled_image: The image to scale.
    :param repeat_list: The number of times a column is repeated.
    :return: The scaled image.
    """
    height, width = unscaled_image.shape
    if width != len(repeat_list):
        raise ValueError("The length of repeat_list should be equal to the width of the image.")
    
    image = image_rotate_cw(unscaled_image)
    scaled_image = _image_scale_up_variable_y(image, repeat_list)
    return image_rotate_ccw(scaled_image)

def image_scale_up_variable(unscaled_image: np.array, repeat_xs: list[int], repeat_ys: list[int]) -> np.array:
    """
    Scale an image by varying the number of times each column/row is repeated.

    If the repeat count is zero, the row/column is removed.

    :param unscaled_image: The image to scale.
    :param repeat_xs: The number of times a column is repeated.
    :param repeat_yw: The number of times a row is repeated.
    :return: The scaled image.
    """
    height, width = unscaled_image.shape
    if width != len(repeat_xs):
        raise ValueError("The length of repeat_xs should be equal to the width of the image.")
    if height != len(repeat_ys):
        raise ValueError("The length of repeat_ys should be equal to the width of the image.")
    
    image = _image_scale_up_variable_x(unscaled_image, repeat_xs)
    image = _image_scale_up_variable_y(image, repeat_ys)
    return image

def image_scale_up_with_noise(unscaled_image: np.array, x_scale: int, y_scale: int, min_noise_count: int, max_noise_count: int, noise_color: int, seed: int) -> np.array:
    """
    Scale up an image and add noise to each cell.

    :param unscaled_image: The image to scale.
    :param x_scale: The factor to scale by in the x direction.
    :param y_scale: The factor to scale by in the y direction.
    :param min_noise_count: The minimum number of noise pixels per cell. Must be 0 or greater.
    :param max_noise_count: The maximum number of noise pixels per cell. Must be less than the number of pixels in a cell.
    :param noise_color: The color of the noise pixels.
    :return: The scaled image.
    """
    if min_noise_count > max_noise_count:
        raise ValueError("min_noise_count <= max_noise_count, is not satisfied")
    if min_noise_count < 0:
        raise ValueError("min_noise_count must be 0 or greater")

    if max_noise_count < 1:
        raise ValueError("number_of_noise_pixels_per_cell must be 1 or greater")
    if max_noise_count >= (x_scale * y_scale):
        raise ValueError("number_of_noise_pixels_per_cell must be less than the number of pixels in a cell.")

    all_positions_inside_cell = []
    for y_offset in range(y_scale):
        for x_offset in range(x_scale):
            all_positions_inside_cell.append((x_offset, y_offset))

    input_image, output_image = image_scale(unscaled_image, 'up', x_scale, 'up', y_scale)
    result_image = output_image.copy()

    height, width = unscaled_image.shape
    for y in range(height):
        for x in range(width):
            iteration_seed = seed + y * 100 + x
            random.Random(iteration_seed + 1).shuffle(all_positions_inside_cell)
            noise_count = random.Random(iteration_seed + 2).randint(min_noise_count, max_noise_count)
            noise_positions = all_positions_inside_cell[:noise_count]
            for dx, dy in noise_positions:
                result_image[y*y_scale + dy, x*x_scale + dx] = noise_color

    return result_image

