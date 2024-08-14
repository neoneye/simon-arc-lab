from typing import Tuple
from enum import Enum
import numpy as np
import random
from .image_util import *

class ImageSymmetryPatternId(Enum):
    HSTACK2 = 'hstack2'
    HSTACK3 = 'hstack3'
    VSTACK2 = 'vstack2'
    VSTACK3 = 'vstack3'
    GRID2X2 = '2x2'

class ImageSymmetry:
    MAX_NUMBER_OF_IMAGES_USED = 4

    PATTERN_IDS = [
        ImageSymmetryPatternId.HSTACK2, 
        ImageSymmetryPatternId.HSTACK3, 
        ImageSymmetryPatternId.VSTACK2, 
        ImageSymmetryPatternId.VSTACK3, 
        ImageSymmetryPatternId.GRID2X2
    ]

    def __init__(self, pattern: ImageSymmetryPatternId):
        self.pattern = pattern

        self.name_image_original = 'orig'
        self.name_image_flipx = 'flipx'
        self.name_image_flipy = 'flipy'
        self.name_image_180 = '180'

        # by default use the original image for all images in the symmetry
        self.name_image_list = []
        for _ in range(ImageSymmetry.MAX_NUMBER_OF_IMAGES_USED):
            name_image = self.name_image_original
            self.name_image_list.append(name_image)

    @classmethod
    def create_random(cls, seed: int) -> Tuple['ImageSymmetry', str]:
        pattern = random.Random(seed).choice(ImageSymmetry.PATTERN_IDS)
        return ImageSymmetry(pattern)
    
    def use_flipx_for_name_image_index(self, index: int):
        self.name_image_list[index] = self.name_image_flipx

    def use_flipy_for_name_image_index(self, index: int):
        self.name_image_list[index] = self.name_image_flipy

    def use_180_for_name_image_index(self, index: int):
        self.name_image_list[index] = self.name_image_180

    def randomize_name_image_list(self, seed: int):
        name_image_map = [
            self.name_image_original,
            self.name_image_flipx,
            self.name_image_flipy,
            self.name_image_180,
        ]
        self.name_image_list = []
        for i in range(ImageSymmetry.MAX_NUMBER_OF_IMAGES_USED):
            name_image = random.Random(seed+i).choice(name_image_map)
            self.name_image_list.append(name_image)

    def execute(self, image: np.array) -> Tuple[np.array, str]:
        image_original = image.copy()
        image_fx = image_flipx(image)
        image_fy = image_flipy(image)
        image_180 = image_rotate_180(image)

        name_to_image = {
            'orig': image_original,
            'flipx': image_fx,
            'flipy': image_fy,
            '180': image_180
        }

        pattern = self.pattern
        name0 = self.name_image_list[0]
        name1 = self.name_image_list[1]
        name2 = self.name_image_list[2]
        name3 = self.name_image_list[3]

        image0 = name_to_image[name0]
        image1 = name_to_image[name1]
        image2 = name_to_image[name2]
        image3 = name_to_image[name3]

        output_image = None
        instruction_sequence = None
        if pattern == ImageSymmetryPatternId.HSTACK2:
            output_image = np.hstack([image0, image1])
            instruction_sequence = f'hstack({name0} {name1})'
        if pattern == ImageSymmetryPatternId.HSTACK3:
            output_image = np.hstack([image0, image1, image2])
            instruction_sequence = f'hstack({name0} {name1} {name2})'
        elif pattern == ImageSymmetryPatternId.VSTACK2:
            output_image = np.vstack([image0, image1])
            instruction_sequence = f'vstack({name0} {name1})'
        elif pattern == ImageSymmetryPatternId.VSTACK3:
            output_image = np.vstack([image0, image1, image2])
            instruction_sequence = f'vstack({name0} {name1} {name2})'
        elif pattern == ImageSymmetryPatternId.GRID2X2:
            output_image = np.vstack([np.hstack([image0, image1]), np.hstack([image2, image3])])
            instruction_sequence = f'2x2({name0} {name1} {name2} {name3})'
        
        return (output_image, instruction_sequence)
