import numpy as np

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
