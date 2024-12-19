import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from .image_compress import *
from .find_bounding_box import find_bounding_box_ignoring_color
from .image_util import *
from .rectangle import Rectangle
from .image_string_representation import image_to_string

class Transformation(Enum):
    ROTATE_CW = 'rotate_cw'
    ROTATE_CCW = 'rotate_ccw'
    ROTATE_180 = 'rotate_180'
    FLIP_X = 'flip_x'
    FLIP_Y = 'flip_y'
    FLIP_A = 'flip_a'
    FLIP_B = 'flip_b'

    @classmethod
    def create_transformation_image_list(cls, image: np.array) -> list[Tuple['Transformation', np.array]]:
        transformation_pairs = [
            (Transformation.ROTATE_CW, image_rotate_cw(image)),
            (Transformation.ROTATE_CCW, image_rotate_ccw(image)),
            (Transformation.ROTATE_180, image_rotate_180(image)),
            (Transformation.FLIP_X, image_flipx(image)),
            (Transformation.FLIP_Y, image_flipy(image)),
            (Transformation.FLIP_A, image_flip_diagonal_a(image)),
            (Transformation.FLIP_B, image_flip_diagonal_b(image))
        ]
        return transformation_pairs

    @classmethod
    def matching_transformations_set(cls, image: np.array, transformation_image_list: Tuple['Transformation', np.array]) -> set['Transformation']:
        transformations = set()
        for transformation, transformed_image in transformation_image_list:
            if np.array_equal(transformed_image, image):
                transformations.add(transformation)
        return transformations

    @classmethod
    def find(cls, image: np.array) -> set['Transformation']:
        pairs = Transformation.create_transformation_image_list(image)
        return Transformation.matching_transformations_set(image, pairs)

class TransformationUtil:
    TRANSFORMATION_ALL_SET = {
        Transformation.ROTATE_CW, 
        Transformation.ROTATE_CCW, 
        Transformation.ROTATE_180, 
        Transformation.FLIP_X, 
        Transformation.FLIP_Y, 
        Transformation.FLIP_A, 
        Transformation.FLIP_B
    }

    @classmethod
    def format_transformation_set(cls, transformation_set: set[Transformation]) -> str:
        if transformation_set == cls.TRANSFORMATION_ALL_SET:
            return 'all'
        if transformation_set == set():
            return 'none'
        transformation_list = [transformation.value for transformation in transformation_set]
        transformation_list.sort()
        return ','.join(transformation_list)


@dataclass
class ShapeCatalogItem:
    long_name: str
    short_name: str
    image: np.array
    transformations: set[Transformation]

class ShapeCatalog:
    def __init__(self):
        self.shapes = []
        self.shapestring_to_index = {}
        self.transformations = set()

    def add(self, long_name: str, short_name: str, pixels: list[list[int]]):
        image = np.array(pixels, dtype=np.uint8)

        pairs = Transformation.create_transformation_image_list(image)
        transformations = Transformation.matching_transformations_set(image, pairs)

        shape = ShapeCatalogItem(long_name, short_name, image, transformations)

        shape_index = len(self.shapes)
        self.shapes.append(shape)

        keys = [
            image_to_string(image)
        ]
        for pair_transformation, pair_image in pairs:
            key = image_to_string(pair_image)
            keys.append(key)
        for key in keys:
            self.shapestring_to_index[key] = shape_index

    def populate(self):
        self.add('plus', '+', [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]])

        self.add('L shape', 'L', [
            [1, 0],
            [1, 1]])

        self.add('hollow square', '□', [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]])

        self.add('square cup', '⊔', [
            [1, 0, 1],
            [1, 1, 1]])

        self.add('H shape', 'H', [
            [1, 0, 1],
            [1, 1, 1],
            [1, 0, 1]])

        self.add('h shape', 'h', [
            [1, 0, 0],
            [1, 1, 1],
            [1, 0, 1]])

class Shape(ABC):
    @abstractmethod
    def shape_type(self) -> str:
        pass

class SimpleShape(Shape):
    def __init__(self, rectangle: Rectangle, shape: ShapeCatalogItem, scale_x: Optional[int], scale_y: Optional[int], transformation_set: set[Transformation]):
        self.rectangle = rectangle
        self.shape = shape
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.transformation_set = transformation_set

    def shape_type(self) -> str:
        return 'SimpleShape'

    def transformation_string(self) -> str:
        return TransformationUtil.format_transformation_set(self.transformation_set)
    
    def scale_mode(self) -> str:
        if self.scale_x is None or self.scale_y is None:
            return 'none'
        if self.scale_x == self.scale_y:
            return f'{self.scale_x}'
        return f'{self.scale_x}x{self.scale_y}'
    
    def __str__(self) -> str:
        return f'{self.shape.long_name} {self.scale_mode()} {self.transformation_string()} {self.rectangle}'

class SolidRectangleShape(Shape):
    def __init__(self, rectangle: Rectangle):
        self.rectangle = rectangle

    def shape_type(self) -> str:
        return 'SolidRectangleShape'
    
    def __str__(self) -> str:
        return f'rectangle {self.rectangle}'

def image_find_shape(image: np.array, verbose: bool = False) -> Optional[Shape]:
    rect = find_bounding_box_ignoring_color(image, 0)
    image_cropped = image[rect.y:rect.y + rect.height, rect.x:rect.x + rect.width]

    transformation_set = Transformation.find(image_cropped)

    compressed_image = image_compress_xy(image_cropped)

    compressed_height, compressed_width = compressed_image.shape

    if verbose:
        print(f'rect: {rect}')
        print(f'compressed_image: {compressed_image}')
        print(f'compressed_height: {compressed_height}')
        print(f'compressed_width: {compressed_width}')

    if compressed_height == 0 or compressed_width == 0:
        raise ValueError("Compressed image has zero height or width")
    
    if compressed_height == 1 and compressed_width == 1:
        return SolidRectangleShape(rect)

    if rect.width % compressed_width == 0:
        scale_x = rect.width // compressed_width
    else:
        scale_x = None

    if rect.height % compressed_height == 0:
        scale_y = rect.height // compressed_height
    else:
        scale_y = None

    if verbose:
        print(f'scale_x: {scale_x} scale_y: {scale_y}')
    
    if scale_x is None or scale_y is None:
        pass
    elif scale_x == 1 and scale_y == 1:
        pass
    else:
        scaled_image = np.kron(compressed_image, np.ones((scale_y, scale_x))).astype(np.uint8)
        if np.array_equal(scaled_image, image_cropped) == False:
            scale_x = None
            scale_y = None

    catalog = ShapeCatalog()
    catalog.populate()

    key = image_to_string(compressed_image)
    shape_index = catalog.shapestring_to_index.get(key)
    if shape_index is None:
        if verbose:
            print(compressed_image.tolist())
            print(f"Shape not found in catalog. key={key}")
        return None

    shape = catalog.shapes[shape_index]
    return SimpleShape(rect, shape, scale_x, scale_y, transformation_set)
