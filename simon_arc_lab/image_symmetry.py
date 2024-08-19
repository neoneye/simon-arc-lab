from typing import Tuple
from enum import Enum
import numpy as np
import random
from .image_util import *

class ImageSymmetryPatternId(Enum):
    HSTACK2 = 'hstack2'
    HSTACK3 = 'hstack3'
    HSTACK4 = 'hstack4'
    HSTACK5 = 'hstack5'
    VSTACK2 = 'vstack2'
    VSTACK3 = 'vstack3'
    VSTACK4 = 'vstack4'
    VSTACK5 = 'vstack5'
    GRID2X2 = '2x2'

class ImageSymmetryMutationId(Enum):
    ORIGINAL = 'orig'
    FLIPX = 'flipx'
    FLIPY = 'flipy'
    FLIP_DIAGONAL_A = 'flipa'
    FLIP_DIAGONAL_B = 'flipb'
    ROTATE_180 = '180'
    ROTATE_CW = 'cw'
    ROTATE_CCW = 'ccw'

class ImageSymmetryBase:
    MAX_NUMBER_OF_IMAGES_USED = 5

    PATTERN_IDS = [
        ImageSymmetryPatternId.HSTACK2, 
        ImageSymmetryPatternId.HSTACK3, 
        ImageSymmetryPatternId.HSTACK4, 
        ImageSymmetryPatternId.HSTACK5, 
        ImageSymmetryPatternId.VSTACK2, 
        ImageSymmetryPatternId.VSTACK3, 
        ImageSymmetryPatternId.VSTACK4, 
        ImageSymmetryPatternId.VSTACK5, 
        ImageSymmetryPatternId.GRID2X2
    ]

    def __init__(self, pattern: ImageSymmetryPatternId):
        self.pattern = pattern

        self.available_name_list = self.populate_available_name_list()

        # by default use the original image for all images in the symmetry
        self.name_list = []
        for _ in range(ImageSymmetryRect.MAX_NUMBER_OF_IMAGES_USED):
            name_image = ImageSymmetryMutationId.ORIGINAL
            self.name_list.append(name_image)

    @classmethod
    def create_random(cls, seed: int) -> Tuple['ImageSymmetryRect', str]:
        pattern = random.Random(seed).choice(ImageSymmetryRect.PATTERN_IDS)
        return ImageSymmetryRect(pattern)
    
    def use_mutation_for_index(self, index: int, mutation_id: ImageSymmetryMutationId):
        if mutation_id not in self.available_name_list:
            raise ValueError(f"Mutation id {mutation_id} is not in available_name list: {self.available_name_list}. class: {self.__class__}")
        self.name_list[index] = mutation_id

    def randomize_name_list(self, seed: int):
        self.name_list = []
        for i in range(ImageSymmetryRect.MAX_NUMBER_OF_IMAGES_USED):
            name_image = random.Random(seed+i).choice(self.available_name_list)
            self.name_list.append(name_image)

    def populate_available_name_list(self) -> list[ImageSymmetryMutationId]:
        raise NotImplementedError()

    def populate_name_to_image_dict(self, image: np.array) -> dict:
        raise NotImplementedError()

    def execute(self, image: np.array) -> np.array:
        name_to_image = self.populate_name_to_image_dict(image)

        pattern = self.pattern
        name0 = self.name_list[0]
        name1 = self.name_list[1]
        name2 = self.name_list[2]
        name3 = self.name_list[3]
        name4 = self.name_list[4]

        image0 = name_to_image[name0]
        image1 = name_to_image[name1]
        image2 = name_to_image[name2]
        image3 = name_to_image[name3]
        image4 = name_to_image[name4]

        output_image = None
        if pattern == ImageSymmetryPatternId.HSTACK2:
            output_image = np.hstack([image0, image1])
        elif pattern == ImageSymmetryPatternId.HSTACK3:
            output_image = np.hstack([image0, image1, image2])
        elif pattern == ImageSymmetryPatternId.HSTACK4:
            output_image = np.hstack([image0, image1, image2, image3])
        elif pattern == ImageSymmetryPatternId.HSTACK5:
            output_image = np.hstack([image0, image1, image2, image3, image4])
        elif pattern == ImageSymmetryPatternId.VSTACK2:
            output_image = np.vstack([image0, image1])
        elif pattern == ImageSymmetryPatternId.VSTACK3:
            output_image = np.vstack([image0, image1, image2])
        elif pattern == ImageSymmetryPatternId.VSTACK4:
            output_image = np.vstack([image0, image1, image2, image3])
        elif pattern == ImageSymmetryPatternId.VSTACK5:
            output_image = np.vstack([image0, image1, image2, image3, image4])
        elif pattern == ImageSymmetryPatternId.GRID2X2:
            output_image = np.vstack([np.hstack([image0, image1]), np.hstack([image2, image3])])
        else:
            raise ValueError(f"Unknown ImageSymmetryPatternId: {pattern}")
        
        return output_image

    def instruction_sequence(self) -> str:
        pattern = self.pattern
        name0 = self.name_list[0].value
        name1 = self.name_list[1].value
        name2 = self.name_list[2].value
        name3 = self.name_list[3].value
        name4 = self.name_list[4].value

        instruction_sequence = None
        if pattern == ImageSymmetryPatternId.HSTACK2:
            instruction_sequence = f'hstack({name0} {name1})'
        elif pattern == ImageSymmetryPatternId.HSTACK3:
            instruction_sequence = f'hstack({name0} {name1} {name2})'
        elif pattern == ImageSymmetryPatternId.HSTACK4:
            instruction_sequence = f'hstack({name0} {name1} {name2} {name3})'
        elif pattern == ImageSymmetryPatternId.HSTACK5:
            instruction_sequence = f'hstack({name0} {name1} {name2} {name3} {name4})'
        elif pattern == ImageSymmetryPatternId.VSTACK2:
            instruction_sequence = f'vstack({name0} {name1})'
        elif pattern == ImageSymmetryPatternId.VSTACK3:
            instruction_sequence = f'vstack({name0} {name1} {name2})'
        elif pattern == ImageSymmetryPatternId.VSTACK4:
            instruction_sequence = f'vstack({name0} {name1} {name2} {name3})'
        elif pattern == ImageSymmetryPatternId.VSTACK5:
            instruction_sequence = f'vstack({name0} {name1} {name2} {name3} {name4})'
        elif pattern == ImageSymmetryPatternId.GRID2X2:
            instruction_sequence = f'2x2({name0} {name1} {name2} {name3})'
        else:
            raise ValueError(f"Unknown ImageSymmetryPatternId: {pattern}")
        
        return instruction_sequence

class ImageSymmetryRect(ImageSymmetryBase):
    def populate_available_name_list(self) -> list[ImageSymmetryMutationId]:
        return [
            ImageSymmetryMutationId.ORIGINAL,
            ImageSymmetryMutationId.FLIPX,
            ImageSymmetryMutationId.FLIPY,
            ImageSymmetryMutationId.ROTATE_180,
        ]

    def populate_name_to_image_dict(self, image: np.array) -> dict:
        image_original = image.copy()
        image_fx = image_flipx(image)
        image_fy = image_flipy(image)
        image_180 = image_rotate_180(image)

        name_to_image = {
            ImageSymmetryMutationId.ORIGINAL: image_original,
            ImageSymmetryMutationId.FLIPX: image_fx,
            ImageSymmetryMutationId.FLIPY: image_fy,
            ImageSymmetryMutationId.ROTATE_180: image_180
        }
        return name_to_image


class ImageSymmetrySquare(ImageSymmetryBase):
    def populate_available_name_list(self) -> list[ImageSymmetryMutationId]:
        return [
            ImageSymmetryMutationId.ORIGINAL,
            ImageSymmetryMutationId.FLIPX,
            ImageSymmetryMutationId.FLIPY,
            ImageSymmetryMutationId.FLIP_DIAGONAL_A,
            ImageSymmetryMutationId.FLIP_DIAGONAL_B,
            ImageSymmetryMutationId.ROTATE_180,
            ImageSymmetryMutationId.ROTATE_CW,
            ImageSymmetryMutationId.ROTATE_CCW,
        ]

    def populate_name_to_image_dict(self, image: np.array) -> dict:
        if image.shape[0] != image.shape[1]:
            raise ValueError(f"ImageSymmetrySquare requires square image. Got {image.shape}")
        
        image_original = image.copy()
        image_fx = image_flipx(image)
        image_fy = image_flipy(image)
        image_fa = image_flip_diagonal_a(image)
        image_fb = image_flip_diagonal_b(image)
        image_180 = image_rotate_180(image)
        image_cw = image_rotate_cw(image)
        image_ccw = image_rotate_ccw(image)

        name_to_image = {
            ImageSymmetryMutationId.ORIGINAL: image_original,
            ImageSymmetryMutationId.FLIPX: image_fx,
            ImageSymmetryMutationId.FLIPY: image_fy,
            ImageSymmetryMutationId.FLIP_DIAGONAL_A: image_fa,
            ImageSymmetryMutationId.FLIP_DIAGONAL_B: image_fb,
            ImageSymmetryMutationId.ROTATE_180: image_180,
            ImageSymmetryMutationId.ROTATE_CW: image_cw,
            ImageSymmetryMutationId.ROTATE_CCW: image_ccw
        }
        return name_to_image
