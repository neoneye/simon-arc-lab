import numpy as np
from .histogram import Histogram

class ImageShape3x3Histogram:

    @classmethod
    def number_of_unique_colors(cls, image: np.array) -> np.array:
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
