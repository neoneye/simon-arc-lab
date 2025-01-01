import unittest
import numpy as np
from typing import Optional
from .shape import *

def process(image: np.array, verbose: bool = False) -> Optional[SimpleShape]:
    return image_find_shape(image, verbose)

class TestShapeName(unittest.TestCase):
    def test_10000_transformation_find_plus_shape(self):
        # Arrange
        image = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]], dtype=np.uint8)
        # Act
        actual = Transformation.find(image)
        # Assert
        expected = {Transformation.ROTATE_CW, Transformation.ROTATE_CCW, Transformation.ROTATE_180, Transformation.FLIP_X, Transformation.FLIP_Y, Transformation.FLIP_A, Transformation.FLIP_B}
        self.assertEqual(actual, expected)

    def test_10001_transformation_find_u_shape(self):
        # Arrange
        image = np.array([
            [1, 0, 1],
            [1, 1, 1]], dtype=np.uint8)
        # Act
        actual = Transformation.find(image)
        # Assert
        expected = {Transformation.FLIP_X}
        self.assertEqual(actual, expected)

    def test_10002_transformation_find_tetris_L_shape(self):
        # Arrange
        image = np.array([
            [1, 0],
            [1, 0],
            [1, 1]], dtype=np.uint8)
        # Act
        actual = Transformation.find(image)
        # Assert
        expected = set()
        self.assertEqual(actual, expected)

    def test_10003_transformation_find_I_shape(self):
        # Arrange
        image = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 1]], dtype=np.uint8)
        # Act
        actual = Transformation.find(image)
        # Assert
        expected = {Transformation.FLIP_A}
        self.assertEqual(actual, expected)

    def test_20000_transformationutil_format_transformation_set_all(self):
        actual = TransformationUtil.format_transformation_set({Transformation.ROTATE_CW, Transformation.ROTATE_CCW, Transformation.ROTATE_180, Transformation.FLIP_X, Transformation.FLIP_Y, Transformation.FLIP_A, Transformation.FLIP_B})
        self.assertEqual(actual, 'all')

    def test_20001_transformationutil_format_transformation_set_none(self):
        actual = TransformationUtil.format_transformation_set(set())
        self.assertEqual(actual, 'none')

    def test_20002_transformationutil_format_transformation_set_alphabetic_sorting(self):
        actual = TransformationUtil.format_transformation_set({Transformation.ROTATE_180, Transformation.FLIP_X})
        self.assertEqual(actual, 'flip_x,rotate_180')

    def test_20003_transformationutil_format_transformation_set_alphabetic_sorting(self):
        actual = TransformationUtil.format_transformation_set({Transformation.FLIP_B, Transformation.FLIP_X, Transformation.FLIP_A})
        self.assertEqual(actual, 'flip_a,flip_b,flip_x')

    def test_30000_solid_rectangle(self):
        # Arrange
        image = np.array([[1]], dtype=np.uint8)
        # Act
        actual = process(image)
        # Assert
        self.assertIsInstance(actual, SolidRectangleShape)
        self.assertEqual(actual.rectangle, Rectangle(0, 0, 1, 1))

    def test_30001_solid_rectangle(self):
        # Arrange
        image = np.array([
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0]], dtype=np.uint8)
        # Act
        actual = process(image)
        # Assert
        self.assertIsInstance(actual, SolidRectangleShape)
        self.assertEqual(actual.rectangle, Rectangle(1, 0, 2, 3))

    def test_30002_solid_rectangle(self):
        # Arrange
        image = np.array([
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0]], dtype=np.uint8)
        # Act
        actual = process(image)
        # Assert
        self.assertIsInstance(actual, SolidRectangleShape)
        self.assertEqual(actual.rectangle, Rectangle(4, 0, 4, 2))

    def test_40000_plus_scale1(self):
        # Arrange
        image = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]], dtype=np.uint8)
        # Act
        actual = process(image)
        # Assert
        self.assertIsInstance(actual, SimpleShape)
        self.assertEqual(actual.rectangle, Rectangle(0, 0, 3, 3))
        self.assertEqual(actual.shape.long_name, 'plus')
        self.assertEqual(actual.transformation_string(), 'all')
        self.assertEqual(actual.scale_mode(), '1')

    def test_40001_plus_scale2(self):
        # Arrange
        image = np.array([
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0]], dtype=np.uint8)
        # Act
        actual = process(image)
        # Assert
        self.assertIsInstance(actual, SimpleShape)
        self.assertEqual(actual.rectangle, Rectangle(0, 0, 6, 6))
        self.assertEqual(actual.shape.long_name, 'plus')
        self.assertEqual(actual.transformation_string(), 'all')
        self.assertEqual(actual.scale_mode(), '2')

    def test_40002_plus_scale2width(self):
        # Arrange
        image = np.array([
            [0, 0, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 0, 0]], dtype=np.uint8)
        # Act
        actual = process(image)
        # Assert
        self.assertIsInstance(actual, SimpleShape)
        self.assertEqual(actual.rectangle, Rectangle(0, 0, 6, 3))
        self.assertEqual(actual.shape.long_name, 'plus')
        self.assertEqual(actual.transformation_string(), 'flip_x,flip_y,rotate_180')
        self.assertEqual(actual.scale_mode(), '2x1')

    def test_40003_plus_scale2height(self):
        # Arrange
        image = np.array([
            [0, 1, 0],
            [0, 1, 0],
            [1, 1, 1],
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0]], dtype=np.uint8)
        # Act
        actual = process(image)
        # Assert
        self.assertIsInstance(actual, SimpleShape)
        self.assertEqual(actual.rectangle, Rectangle(0, 0, 3, 6))
        self.assertEqual(actual.shape.long_name, 'plus')
        self.assertEqual(actual.transformation_string(), 'flip_x,flip_y,rotate_180')
        self.assertEqual(actual.scale_mode(), '1x2')

    def test_40002_plus_compressed(self):
        # Arrange
        image = np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]], dtype=np.uint8)
        # Act
        actual = process(image)
        # Assert
        self.assertIsInstance(actual, SimpleShape)
        self.assertEqual(actual.rectangle, Rectangle(0, 0, 5, 5))
        self.assertEqual(actual.shape.long_name, 'plus')
        self.assertEqual(actual.transformation_string(), 'all')
        self.assertEqual(actual.scale_mode(), 'none')

    def test_40003_plus_compressed(self):
        # Arrange
        image = np.array([
            [0, 1, 0, 0],
            [1, 1, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0]], dtype=np.uint8)
        # Act
        actual = process(image)
        # Assert
        self.assertIsInstance(actual, SimpleShape)
        self.assertEqual(actual.rectangle, Rectangle(0, 0, 4, 5))
        self.assertEqual(actual.shape.long_name, 'plus')
        self.assertEqual(actual.transformation_string(), 'none')
        self.assertEqual(actual.scale_mode(), 'none')

    def test_40004_plus_crop(self):
        # Arrange
        image = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        # Act
        actual = process(image)
        # Assert
        self.assertIsInstance(actual, SimpleShape)
        self.assertEqual(actual.rectangle, Rectangle(1, 2, 4, 4))
        self.assertEqual(actual.shape.long_name, 'plus')
        self.assertEqual(actual.transformation_string(), 'all')
        self.assertEqual(actual.scale_mode(), 'none')

    def test_50000_L_shape(self):
        # Arrange
        image = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        # Act
        actual = process(image)
        # Assert
        self.assertIsInstance(actual, SimpleShape)
        self.assertEqual(actual.rectangle, Rectangle(1, 1, 4, 3))
        self.assertEqual(actual.shape.long_name, 'L shape')
        self.assertEqual(actual.transformation_string(), 'none')
        self.assertEqual(actual.scale_mode(), 'none')

    def test_60000_hollow_square(self):
        # Arrange
        image = np.array([
            [0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 1, 0, 0, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        # Act
        actual = process(image)
        # Assert
        self.assertIsInstance(actual, SimpleShape)
        self.assertEqual(actual.rectangle, Rectangle(1, 0, 4, 4))
        self.assertEqual(actual.shape.long_name, 'hollow square')
        self.assertEqual(actual.transformation_string(), 'flip_x')
        self.assertEqual(actual.scale_mode(), 'none')

    def test_70000_square_cup(self):
        # Arrange
        image = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 1, 0],
            [0, 1, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        # Act
        actual = process(image)
        # Assert
        self.assertIsInstance(actual, SimpleShape)
        self.assertEqual(actual.rectangle, Rectangle(1, 1, 4, 3))
        self.assertEqual(actual.shape.long_name, 'square cup')
        self.assertEqual(actual.transformation_string(), 'none')
        self.assertEqual(actual.scale_mode(), 'none')

    def test_80000_H_shape(self):
        # Arrange
        image = np.array([
            [1, 0, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 1, 1],
            [1, 0, 1, 1]], dtype=np.uint8)
        # Act
        actual = process(image)
        # Assert
        self.assertIsInstance(actual, SimpleShape)
        self.assertEqual(actual.rectangle, Rectangle(0, 0, 4, 4))
        self.assertEqual(actual.shape.long_name, 'H shape')
        self.assertEqual(actual.transformation_string(), 'none')
        self.assertEqual(actual.scale_mode(), 'none')

    def test_90000_h_shape(self):
        # Arrange
        image = np.array([
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 0, 1, 1]], dtype=np.uint8)
        # Act
        actual = process(image)
        # Assert
        self.assertIsInstance(actual, SimpleShape)
        self.assertEqual(actual.rectangle, Rectangle(0, 0, 4, 4))
        self.assertEqual(actual.shape.long_name, 'h shape')
        self.assertEqual(actual.transformation_string(), 'none')
        self.assertEqual(actual.scale_mode(), 'none')

    def test_90001_h_shape_rot_ccw(self):
        # Arrange
        image = np.array([
            [0, 1, 1],
            [0, 1, 0],
            [1, 1, 1]], dtype=np.uint8)
        # Act
        actual = process(image)
        # Assert
        self.assertIsInstance(actual, SimpleShape)
        self.assertEqual(actual.rectangle, Rectangle(0, 0, 3, 3))
        self.assertEqual(actual.shape.long_name, 'h shape')
        self.assertEqual(actual.transformation_string(), 'none')
        self.assertEqual(actual.scale_mode(), '1')

    def test_90002_h_shape_rot_cw(self):
        # Arrange
        image = np.array([
            [1, 1, 1],
            [0, 1, 0],
            [1, 1, 0]], dtype=np.uint8)
        # Act
        actual = process(image)
        # Assert
        self.assertIsInstance(actual, SimpleShape)
        self.assertEqual(actual.rectangle, Rectangle(0, 0, 3, 3))
        self.assertEqual(actual.shape.long_name, 'h shape')
        self.assertEqual(actual.transformation_string(), 'none')
        self.assertEqual(actual.scale_mode(), '1')

    def test_90003_h_shape_rot_180(self):
        # Arrange
        image = np.array([
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 1]], dtype=np.uint8)
        # Act
        actual = process(image)
        # Assert
        self.assertIsInstance(actual, SimpleShape)
        self.assertEqual(actual.rectangle, Rectangle(0, 0, 3, 3))
        self.assertEqual(actual.shape.long_name, 'h shape')
        self.assertEqual(actual.transformation_string(), 'none')
        self.assertEqual(actual.scale_mode(), '1')

    def test_100001_tetris_skew_tetromino(self):
        # Arrange
        image = np.array([
            [1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        # Act
        actual = process(image)
        # Assert
        self.assertIsInstance(actual, SimpleShape)
        self.assertEqual(actual.rectangle, Rectangle(0, 0, 5, 4))
        self.assertEqual(actual.shape.long_name, 'skew tetromino')
        self.assertEqual(actual.transformation_string(), 'none')
        self.assertEqual(actual.scale_mode(), 'none')

    def test_110000_t_shape(self):
        # Arrange
        image = np.array([
            [1, 1, 1, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 0]], dtype=np.uint8)
        # Act
        actual = process(image)
        # Assert
        self.assertIsInstance(actual, SimpleShape)
        self.assertEqual(actual.rectangle, Rectangle(0, 0, 4, 3))
        self.assertEqual(actual.shape.long_name, 'T shape')
        self.assertEqual(actual.transformation_string(), 'none')
        self.assertEqual(actual.scale_mode(), 'none')

    def test_120000_x_shape(self):
        # Arrange
        image = np.array([
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [0, 0, 1, 0, 0],
            [1, 1, 0, 1, 1]], dtype=np.uint8)
        # Act
        actual = process(image)
        # Assert
        self.assertIsInstance(actual, SimpleShape)
        self.assertEqual(actual.rectangle, Rectangle(0, 0, 5, 4))
        self.assertEqual(actual.shape.long_name, 'X shape')
        self.assertEqual(actual.transformation_string(), 'flip_x')
        self.assertEqual(actual.scale_mode(), 'none')

    def test_130000_y_shape(self):
        # Arrange
        image = np.array([
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1]], dtype=np.uint8)
        # Act
        actual = process(image)
        # Assert
        self.assertIsInstance(actual, SimpleShape)
        self.assertEqual(actual.rectangle, Rectangle(0, 0, 5, 5))
        self.assertEqual(actual.shape.long_name, 'y shape')
        self.assertEqual(actual.transformation_string(), 'flip_b')
        self.assertEqual(actual.scale_mode(), 'none')

    def test_140000_y_shape(self):
        # Arrange
        image = np.array([
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1]], dtype=np.uint8)
        # Act
        actual = process(image)
        # Assert
        self.assertIsInstance(actual, SimpleShape)
        self.assertEqual(actual.rectangle, Rectangle(0, 0, 5, 5))
        self.assertEqual(actual.shape.long_name, '1left-3-1right shape')
        self.assertEqual(actual.transformation_string(), 'rotate_180')
        self.assertEqual(actual.scale_mode(), 'none')

    def test_150000_openboxa_shape(self):
        # Arrange
        image = np.array([
            [0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]], dtype=np.uint8)
        # Act
        actual = process(image)
        # Assert
        self.assertIsInstance(actual, SimpleShape)
        self.assertEqual(actual.rectangle, Rectangle(0, 1, 5, 5))
        self.assertEqual(actual.shape.long_name, 'open-box-a')
        self.assertEqual(actual.transformation_string(), 'flip_x')
        self.assertEqual(actual.scale_mode(), 'none')

    def test_160000_openboxb_shape(self):
        # Arrange
        image = np.array([
            [0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]], dtype=np.uint8)
        # Act
        actual = process(image)
        # Assert
        self.assertIsInstance(actual, SimpleShape)
        self.assertEqual(actual.rectangle, Rectangle(0, 1, 5, 5))
        self.assertEqual(actual.shape.long_name, 'open-box-b')
        self.assertEqual(actual.transformation_string(), 'none')
        self.assertEqual(actual.scale_mode(), 'none')

    def test_170000_e_shape(self):
        # Arrange
        image = np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1]], dtype=np.uint8)
        # Act
        actual = process(image)
        # Assert
        self.assertIsInstance(actual, SimpleShape)
        self.assertEqual(actual.rectangle, Rectangle(0, 0, 5, 5))
        self.assertEqual(actual.shape.long_name, 'E shape')
        self.assertEqual(actual.transformation_string(), 'flip_y')
        self.assertEqual(actual.scale_mode(), 'none')
