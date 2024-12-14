import unittest
import numpy as np
from .image_gaussian_splatting import ImageGaussianSplatting

class TestImageGaussianSplatting(unittest.TestCase):
    def test_10000_empty(self):
        # Arrange
        input = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        # Act
        igs = ImageGaussianSplatting(input)
        # Assert
        self.assertTrue(np.isnan(igs.angle), f"angle is not NaN, got {igs.angle}")
        self.assertTrue(np.isnan(igs.x_c), f"x_c is not NaN, got {igs.x_c}")
        self.assertTrue(np.isnan(igs.y_c), f"y_c is not NaN, got {igs.y_c}")
        self.assertTrue(np.isnan(igs.spread_primary), f"spread_primary is not NaN, got {igs.spread_primary}")
        self.assertTrue(np.isnan(igs.spread_secondary), f"spread_secondary is not NaN, got {igs.spread_secondary}")

    def test_20000_one_point(self):
        # Arrange
        input = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        # Act
        igs = ImageGaussianSplatting(input)
        # Assert
        self.assertAlmostEqual(igs.angle, 0.0, places=2)
        self.assertAlmostEqual(igs.x_c, 2.0, places=2)
        self.assertAlmostEqual(igs.y_c, 2.0, places=2)
        self.assertAlmostEqual(igs.spread_primary, 0.0, places=2)
        self.assertAlmostEqual(igs.spread_secondary, 0.0, places=2)

    def test_30000_multiple_points_diagonal(self):
        # Arrange
        input = np.array([
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        # Act
        igs = ImageGaussianSplatting(input)
        # Assert
        self.assertAlmostEqual(igs.angle, np.deg2rad(135), places=2)
        self.assertAlmostEqual(igs.x_c, 3.0, places=2)
        self.assertAlmostEqual(igs.y_c, 1.0, places=2)
        self.assertAlmostEqual(igs.spread_primary, 1.41, places=2)
        self.assertAlmostEqual(igs.spread_secondary, 0.0, places=2)

    def test_30001_multiple_points_diagonal(self):
        # Arrange
        input = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0]], dtype=np.uint8)
        # Act
        igs = ImageGaussianSplatting(input)
        # Assert
        self.assertAlmostEqual(igs.angle, np.deg2rad(45), places=2)
        self.assertAlmostEqual(igs.x_c, 1.5, places=2)
        self.assertAlmostEqual(igs.y_c, 1.5, places=2)
        self.assertAlmostEqual(igs.spread_primary, 1.83, places=2)
        self.assertAlmostEqual(igs.spread_secondary, 0.0, places=2)

    def test_30002_multiple_points(self):
        # Arrange
        input = np.array([
            [1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1]], dtype=np.uint8)
        # Act
        igs = ImageGaussianSplatting(input)
        # Assert
        self.assertAlmostEqual(igs.angle, -np.deg2rad(138), places=2)
        self.assertAlmostEqual(igs.x_c, 2.0, places=2)
        self.assertAlmostEqual(igs.y_c, 1.5, places=2)
        self.assertAlmostEqual(igs.spread_primary, 1.73, places=2)
        self.assertAlmostEqual(igs.spread_secondary, 0.37, places=2)
