from typing import Dict
import numpy as np

def image_create(width: int, height: int, color: int) -> np.array:
    image = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            image[y, x] = color
    return image

def image_rotate_cw(image: np.array) -> np.array:
    return np.rot90(image, k=-1)

def image_rotate_ccw(image: np.array) -> np.array:
    return np.rot90(image)

def image_rotate_180(image: np.array) -> np.array:
    return np.rot90(image, k=2)

def count_same_color_as_center_with_one_neighbor_nowrap(image: np.array, dx: int, dy: int) -> np.array:
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

def count_neighbors_with_same_color_nowrap(image: np.array) -> np.array:
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

def pixels_with_k_matching_neighbors_nowrap(image: np.array, k: int) -> np.array:
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

def all_neighbors_matching_center_nowrap(image: np.array) -> np.array:
    """
    Checks if the center pixel has the same color as all 8 surrounding pixels.
    
    Sets the value to 1 if all surrounding pixels have the same color as the center pixel.
    Sets the value to 0 if any surrounding pixel differs in color from the center pixel.
    
    :param image: 2D numpy array representing the image.
    :return: 2D numpy array of the same size as the input image, with each element set to 1 if 
             all surrounding pixels have the same color as the corresponding center pixel, otherwise 0.
    """
    return pixels_with_k_matching_neighbors_nowrap(image, 8)    

def compress_x(image: np.array) -> np.array:
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

def compress_y(image: np.array) -> np.array:
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

def compress_xy(image: np.array) -> np.array:
    """
    Eliminate adjacent duplicate rows+columns
    """
    return compress_y(compress_x(image))

def image_translate_wrap(image: np.array, dx: int, dy: int) -> np.array:
    """
    Move pixels by dx, dy, wrapping around the image.

    :param image: The image to process.
    :param dx: The horizontal translation.
    :param dy: The vertical translation.
    :return: An image of the same size as the input image.
    """
    if dx == 0 and dy == 0:
        raise ValueError("dx and dy cannot both be zero.")

    height, width = image.shape
    new_image = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            new_y = (height + y + dy) % height
            new_x = (width + x + dx) % width
            new_image[new_y, new_x] = image[y, x]

    return new_image

def image_replace_colors(image: np.array, color_mapping: Dict[int, int]) -> np.array:
    """
    Replace colors in an image according to a dictionary.

    :param image: The image to process.
    :param color_mapping: A dictionary where the keys are the colors to replace and the values are the new colors.
    :return: An image of the same size as the input image.
    """
    new_image = np.copy(image)
    
    for old_color, new_color in color_mapping.items():
        mask = image == old_color
        new_image[mask] = new_color
        
    return new_image

def image_get_row_as_list(image: np.array, row_index: int) -> list[int]:
    """
    Get a row from an image.

    :param image: The image to process.
    :param row_index: The index of the row to get.
    :return: The row as a list.
    """
    height = image.shape[0]
    if row_index < 0 or row_index >= height:
        raise ValueError(f"Row index {row_index} is out of bounds for image with height {height}")
    return list(image[row_index])

def image_get_column_as_list(image: np.array, column_index: int) -> list[int]:
    """
    Get a column from an image.

    :param image: The image to process.
    :param column_index: The index of the column to get.
    :return: The column as a list.
    """
    width = image.shape[1]
    if column_index < 0 or column_index >= width:
        raise ValueError(f"Column index {column_index} is out of bounds for image with width {width}")
    return list(image[:, column_index])
