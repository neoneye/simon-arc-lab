import numpy as np
from collections import Counter

class ImageShape2x2:
    # horizontal
    TOPLEFT_EQUAL_TOPRIGHT = 1
    BOTTOMLEFT_EQUAL_BOTTOMRIGHT = 2

    # vertical
    TOPLEFT_EQUAL_BOTTOMLEFT = 4
    TOPRIGHT_EQUAL_BOTTOMRIGHT = 8

    # diagonal
    TOPLEFT_EQUAL_BOTTOMRIGHT = 16
    TOPRIGHT_EQUAL_BOTTOMLEFT = 32

    @classmethod
    def apply(cls, image: np.array) -> np.array:
        """
        Compare pixel colors in a 2x2 neighborhood.

        There are 6 types of comparisons, represented as 6 bits.
        A bit is set when the comparison is satisfied.

        When the entire mask is zero, it means no neighbors have the same color.

        :param image: image
        return: image with the size (width+1) x (height+1)
        """

        height, width = image.shape

        # border 1px with color 255
        image_with_border = np.pad(image, pad_width=1, mode='constant', constant_values=255)

        new_image = np.zeros((height + 1, width + 1), dtype=np.uint8)
        for y in range(height + 1):
            for x in range(width + 1):
                value_topleft = image_with_border[y, x]
                value_topright = image_with_border[y, x+1]
                value_bottomleft = image_with_border[y+1, x]
                value_bottomright = image_with_border[y+1, x+1]

                mask = 0

                if value_topleft == value_topright:
                    mask |= cls.TOPLEFT_EQUAL_TOPRIGHT
                if value_bottomleft == value_bottomright:
                    mask |= cls.BOTTOMLEFT_EQUAL_BOTTOMRIGHT
                if value_topleft == value_bottomleft:
                    mask |= cls.TOPLEFT_EQUAL_BOTTOMLEFT
                if value_topright == value_bottomright:
                    mask |= cls.TOPRIGHT_EQUAL_BOTTOMRIGHT
                if value_topleft == value_bottomright:
                    mask |= cls.TOPLEFT_EQUAL_BOTTOMRIGHT
                if value_topright == value_bottomleft:
                    mask |= cls.TOPRIGHT_EQUAL_BOTTOMLEFT
                
                new_image[y, x] = mask
        
        return new_image

    @classmethod
    def shape_id_list(cls, image: np.array) -> list[int]:
        """
        Extract the shape ids that are present in the image.

        :param image: The image to analyze.
        return: list of shape ids, eg. [3, 12, 22, 25, 37, 42, 63]
        """
        shapeid_image = cls.apply(image)

        # histogram of the shape ids, and the count of each shape id
        counter = Counter(shapeid_image.flatten())
        # extract the shape ids and ignore the count
        shape_ids = list(counter.keys())
        # cast from np uint to int
        shape_ids = [int(key) for key in shape_ids]
        shape_ids = sorted(shape_ids)
        return shape_ids
