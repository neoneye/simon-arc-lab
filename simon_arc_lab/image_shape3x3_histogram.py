import numpy as np
from .histogram import Histogram

class ImageShape3x3Histogram:

    @classmethod
    def number_of_unique_colors_all9(cls, image: np.array) -> np.array:
        """
        Count the number of unique colors inside a 3x3 neighborhood, all 9 pixels.

        When all the 9 pixels use different colors, the number of unique colors is 9.

        When all the 9 pixels use the same color, the number of unique colors is 1.

        Pixes that are outside the image are not counted.

        :param image: image
        return: image with the same size as the input image
        """

        height, width = image.shape
        new_image = np.zeros((height, width), dtype=np.uint8)
        
        for y in range(height):
            for x in range(width):
                histogram = Histogram.empty()
                
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            histogram.increment(image[ny, nx])
                
                new_image[y, x] = histogram.number_of_unique_colors()
        
        return new_image

    @classmethod
    def number_of_unique_colors_around_center(cls, image: np.array) -> np.array:
        """
        Count the number of unique colors inside a 3x3 neighborhood, except the center, so 8 pixels.

        When all the 8 pixels use different colors, the number of unique colors is 8.

        When all the 8 pixels use the same color, the number of unique colors is 1.

        Pixes that are outside the image are not counted.

        :param image: image
        return: image with the same size as the input image
        """

        height, width = image.shape
        new_image = np.zeros((height, width), dtype=np.uint8)
        
        for y in range(height):
            for x in range(width):
                histogram = Histogram.empty()
                
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            histogram.increment(image[ny, nx])
                
                new_image[y, x] = histogram.number_of_unique_colors()
        
        return new_image

    @classmethod
    def number_of_unique_colors_in_diamond4(cls, image: np.array) -> np.array:
        """
        Count the number of unique colors in the 4 pixels adjacent to the center pixel. 

        When all the 4 pixels use different colors, the number of unique colors is 4.

        When all the 4 pixels use the same color, the number of unique colors is 1.

        Pixes that are outside the image are not counted.

        :param image: image
        return: image with the same size as the input image
        """

        height, width = image.shape
        new_image = np.zeros((height, width), dtype=np.uint8)
        
        for y in range(height):
            for x in range(width):
                dx_dy_list = [(0, -1), (-1, 0), (1, 0), (0, 1)]
                histogram = Histogram.empty()
                
                for dx_dy in dx_dy_list:
                    dx = dx_dy[0]
                    dy = dx_dy[1]
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        histogram.increment(image[ny, nx])
                
                new_image[y, x] = histogram.number_of_unique_colors()
        
        return new_image

    @classmethod
    def number_of_unique_colors_in_diamond5(cls, image: np.array) -> np.array:
        """
        Count the number of unique colors inside a 3x3 neighborhood, except the corners, so 5 pixels.

        When all the 5 pixels use different colors, the number of unique colors is 5.

        When all the 5 pixels use the same color, the number of unique colors is 1.

        Pixes that are outside the image are not counted.

        :param image: image
        return: image with the same size as the input image
        """

        height, width = image.shape
        new_image = np.zeros((height, width), dtype=np.uint8)
        
        for y in range(height):
            for x in range(width):
                dx_dy_list = [(0, -1), (-1, 0), (0, 0), (1, 0), (0, 1)]
                histogram = Histogram.empty()
                
                for dx_dy in dx_dy_list:
                    dx = dx_dy[0]
                    dy = dx_dy[1]
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        histogram.increment(image[ny, nx])
                
                new_image[y, x] = histogram.number_of_unique_colors()
        
        return new_image
