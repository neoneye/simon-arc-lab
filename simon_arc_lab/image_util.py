import numpy as np
import random

def image_create(width, height, color):
    image = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            image[y, x] = color
    return image

def histogram_of_image(image):
    hist = {}
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            color = image[y, x]
            if color in hist:
                hist[color] += 1
            else:
                hist[color] = 1
    return hist

def sorted_histogram_of_image(image):
    hist = histogram_of_image(image)
    # sort by popularity, if there is a tie, sort by color
    items = sorted(hist.items(), key=lambda item: (-item[1], item[0]))
    return items

def pretty_histogram_of_image(image):
    hist = sorted_histogram_of_image(image)
    return ','.join([f'{color}:{count}' for color, count in hist])

def image_create_random_with_two_colors(width, height, color1, color2, ratio, seed):
    image = image_create(width, height, color1)

    positions = []
    for y in range(height):
        for x in range(width):
            positions += [(y, x)]

    random.Random(seed).shuffle(positions)

    # take a ratio of the positions
    num_positions = int(len(positions) * ratio)
    for i in range(num_positions):
        y, x = positions[i]
        image[y, x] = color2
    return image

def image_create_random_with_three_colors(width, height, color0, color1, color2, weight0, weight1, weight2, seed):
    image = image_create(width, height, color0)

    positions = [(y, x) for y in range(height) for x in range(width)]

    random.Random(seed).shuffle(positions)

    total_weight = weight0 + weight1 + weight2
    num_positions_a = int(len(positions) * (weight1 / total_weight))
    num_positions_b = int(len(positions) * (weight2 / total_weight))

    for i in range(num_positions_a):
        y, x = positions[i]
        image[y, x] = color1

    for i in range(num_positions_b):
        y, x = positions[i + num_positions_a]
        image[y, x] = color2

    return image

def image_create_random_with_four_colors(width, height, color0, color1, color2, color3, weight0, weight1, weight2, weight3, seed):
    image = image_create(width, height, color0)

    positions = [(y, x) for y in range(height) for x in range(width)]

    random.Random(seed).shuffle(positions)

    total_weight = weight0 + weight1 + weight2 + weight3
    num_positions_a = int(len(positions) * (weight1 / total_weight))
    num_positions_b = int(len(positions) * (weight2 / total_weight))
    num_positions_c = int(len(positions) * (weight3 / total_weight))

    for i in range(num_positions_a):
        y, x = positions[i]
        image[y, x] = color1

    for i in range(num_positions_b):
        y, x = positions[i + num_positions_a]
        image[y, x] = color2

    for i in range(num_positions_c):
        y, x = positions[i + num_positions_a + num_positions_b]
        image[y, x] = color3

    return image

def image_rotate_cw(image):
    return np.rot90(image, k=-1)

def image_rotate_ccw(image):
    return np.rot90(image)

def image_rotate_180(image):
    return np.rot90(image, k=2)

def bresenham_line(image, x0, y0, x1, y1, color):
    """
    Draw a line on an image using Bresenham's line algorithm.

    https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm

    :param image: The image to draw the line on
    :param x0: The x-coordinate of the start point
    :param y0: The y-coordinate of the start point
    :param x1: The x-coordinate of the end point
    :param y1: The y-coordinate of the end point
    :param color: The color to draw the line with
    :return: The image with the line drawn on it
    """
    height, width = image.shape

    # Check if the coordinates are outside the image bounds
    if not (0 <= x0 < width and 0 <= y0 < height and 0 <= x1 < width and 0 <= y1 < height):
        raise ValueError(f"Coordinates ({x0}, {y0}), ({x1}, {y1}) are outside the image bounds of width {width} and height {height}")

    # Clone the image to avoid mutating the original
    new_image = np.copy(image)

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        new_image[y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return new_image

def count_same_color_as_center_with_one_neighbor_nowrap(image, dx, dy):
    """
    Is the center pixel the same color as the pixel at dx, dy?

    Sets the value to 1 if the two pixels have the same value.

    Sets the value to 0 if the two pixels have differ.

    :param image: The image to process.
    :return: An image of the same size as the input image.
    """
    if dx == 0 and dy == 0:
        raise ValueError("dx and dy cannot both be zero.")
    
    height, width = image.shape
    result_image = np.zeros((height, width), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            center_color = image[y, x]
            same_color = 0
            
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                if image[ny, nx] == center_color:
                    same_color = 1
            
            result_image[y, x] = same_color
    
    return result_image

def count_neighbors_with_same_color_nowrap(image):
    """
    Counts the number of neighboring pixels with the same color as the center pixel.
    
    The maximum number of neighbors is 8.
    The minimum number of neighbors is 0.
    
    :param image: 2D numpy array representing the image.
    :return: 2D numpy array of the same size as the input image, with each element representing 
             the count of neighbors with the same color as the corresponding center pixel.
    """
    height, width = image.shape
    count_matrix = np.zeros((height, width), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            center_color = image[y, x]
            same_color_count = 0
            
            # Check all 8 neighbors
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if image[ny, nx] == center_color:
                            same_color_count += 1
            
            count_matrix[y, x] = same_color_count
    
    return count_matrix

def pixels_with_k_matching_neighbors_nowrap(image, k):
    """
    Identifies pixels where exactly k neighbors have the same color as the center pixel.
    
    Sets the value to 1 where the condition is satisfied, otherwise sets it to 0.
    
    :param image: 2D numpy array representing the image.
    :param k: The number of neighbors that should have the same color as the center pixel.
    :return: 2D numpy array of the same size as the input image, with each element set to 1 if 
             the corresponding center pixel has exactly k matching neighbors, otherwise 0.
    """

    count_matrix  = count_neighbors_with_same_color_nowrap(image)
    height, width = image.shape
    result_image = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            count = count_matrix[y, x]
            if count == k:
                result_image[y, x] = 1

    return result_image

def all_neighbors_matching_center_nowrap(image):
    """
    Checks if the center pixel has the same color as all 8 surrounding pixels.
    
    Sets the value to 1 if all surrounding pixels have the same color as the center pixel.
    Sets the value to 0 if any surrounding pixel differs in color from the center pixel.
    
    :param image: 2D numpy array representing the image.
    :return: 2D numpy array of the same size as the input image, with each element set to 1 if 
             all surrounding pixels have the same color as the corresponding center pixel, otherwise 0.
    """
    return pixels_with_k_matching_neighbors_nowrap(image, 8)    
