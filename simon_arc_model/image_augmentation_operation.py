"""
IDEA: Normal or Transpose so it's using the smallest RLE compressed representation
IDEA: Scale up by 2, 3
IDEA: Denoise
"""
from enum import Enum
from simon_arc_lab.image_util import *
from simon_arc_lab.image_rotate45 import *
from simon_arc_lab.image_skew import *

class ImageAugmentationOperation(Enum):
    DO_NOTHING = 'do_nothing'
    ROTATE_CW = 'rotate_cw'
    ROTATE_CCW = 'rotate_ccw'
    ROTATE_180 = 'rotate_180'
    FLIP_X = 'flip_x'
    FLIP_Y = 'flip_y'
    FLIP_A = 'flip_a'
    FLIP_B = 'flip_b'
    SKEW_UP = 'skew_up'
    SKEW_DOWN = 'skew_down'
    SKEW_LEFT = 'skew_left'
    SKEW_RIGHT = 'skew_right'
    ROTATE_CW_45 = 'rotate_cw_45'
    ROTATE_CCW_45 = 'rotate_ccw_45'

    def apply(self, image: np.array) -> np.array:
        if self == ImageAugmentationOperation.DO_NOTHING:
            return image
        elif self == ImageAugmentationOperation.ROTATE_CW:
            return image_rotate_cw(image)
        elif self == ImageAugmentationOperation.ROTATE_CCW:
            return image_rotate_ccw(image)
        elif self == ImageAugmentationOperation.ROTATE_180:
            return image_rotate_180(image)
        elif self == ImageAugmentationOperation.FLIP_X:
            return image_flipx(image)
        elif self == ImageAugmentationOperation.FLIP_Y:
            return image_flipy(image)
        elif self == ImageAugmentationOperation.FLIP_A:
            return image_flip_diagonal_a(image)
        elif self == ImageAugmentationOperation.FLIP_B:
            return image_flip_diagonal_b(image)
        elif self == ImageAugmentationOperation.SKEW_UP:
            fill_color = 10
            return image_skew(image, fill_color, SkewDirection.UP)
        elif self == ImageAugmentationOperation.SKEW_DOWN:
            fill_color = 10
            return image_skew(image, fill_color, SkewDirection.DOWN)
        elif self == ImageAugmentationOperation.SKEW_LEFT:
            fill_color = 10
            return image_skew(image, fill_color, SkewDirection.LEFT)
        elif self == ImageAugmentationOperation.SKEW_RIGHT:
            fill_color = 10
            return image_skew(image, fill_color, SkewDirection.RIGHT)
        elif self == ImageAugmentationOperation.ROTATE_CW_45:
            fill_color = 10
            return image_rotate_cw_45(image, fill_color)
        elif self == ImageAugmentationOperation.ROTATE_CCW_45:
            fill_color = 10
            return image_rotate_ccw_45(image, fill_color)
        else:
            raise ValueError(f'Unknown ImageAugmentationOperation: {self}')
