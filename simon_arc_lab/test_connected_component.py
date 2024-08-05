import unittest
import numpy as np
from .connected_component import ConnectedComponent, ConnectedComponentItem
from .pixel_connectivity import PixelConnectivity

class TestConnectedComponent(unittest.TestCase):
    def test_find_objects_neighbors_10000(self):
        # Arrange
        input_image = np.array([
            [5, 5, 5, 5, 5],
            [5, 8, 8, 5, 8],
            [5, 8, 5, 5, 8],
            [5, 5, 5, 5, 8],
        ], dtype=np.uint8)

        # Act
        mask_vec = ConnectedComponent.find_objects(PixelConnectivity.CONNECTIVITY4, input_image)

        # Assert
        self.assertEqual(len(mask_vec), 3)
        output = np.vstack(mask_vec)
        expected = np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 1, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 1, 1, 0],

            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],

            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(output, expected))

    def test_find_objects_neighbors_10001(self):
        # Arrange
        input_image = np.array([
            [5, 5, 5, 5, 5],
            [5, 6, 6, 6, 5],
            [5, 6, 5, 6, 5],
            [5, 6, 6, 6, 5],
            [5, 5, 5, 5, 5],
        ], dtype=np.uint8)

        # Act
        mask_vec = ConnectedComponent.find_objects(PixelConnectivity.CONNECTIVITY4, input_image)

        # Assert
        self.assertEqual(len(mask_vec), 3)
        output = np.vstack(mask_vec)
        expected = np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],

            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],

            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(output, expected))

    def test_find_objects_neighbors_10002(self):
        # Arrange
        input_image = np.array([
            [9, 5, 5],
            [5, 9, 5],
            [5, 5, 9],
        ], dtype=np.uint8)

        # Act
        mask_vec = ConnectedComponent.find_objects(PixelConnectivity.CONNECTIVITY4, input_image)

        # Assert
        self.assertEqual(len(mask_vec), 5)
        output = np.vstack(mask_vec)
        expected = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],

            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0],

            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],

            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],

            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(output, expected))

    def test_find_objects_neighbors_10003(self):
        # Arrange
        input_image = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ], dtype=np.uint8)

        # Act
        mask_vec = ConnectedComponent.find_objects(PixelConnectivity.CONNECTIVITY4, input_image)

        # Assert
        self.assertEqual(len(mask_vec), 2)
        output = np.vstack(mask_vec)
        expected = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],

            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(output, expected))

    def test_find_objects_neighbors_10004(self):
        # Arrange
        input_image = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ], dtype=np.uint8)

        # Act
        mask_vec = ConnectedComponent.find_objects(PixelConnectivity.CONNECTIVITY4, input_image)

        # Assert
        self.assertEqual(len(mask_vec), 2)
        output = np.vstack(mask_vec)
        expected = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],

            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(output, expected))

    def test_find_objects_connectivity4_10005(self):
        # Arrange
        input_image = np.array([
            [7, 7, 7, 0, 0, 5],
            [7, 7, 7, 0, 0, 5],
            [7, 7, 7, 0, 0, 5],
            [8, 8, 8, 8, 5, 8],
            [8, 8, 8, 5, 8, 8],
            [8, 7, 8, 8, 8, 8],
            [7, 7, 7, 8, 8, 8],
        ], dtype=np.uint8)

        # Act
        actual = ConnectedComponent.find_objects(PixelConnectivity.CONNECTIVITY4, input_image)

        # Assert
        self.assertEqual(len(actual), 7)

        expected0 = np.array([
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual[0], expected0))

        expected1 = np.array([
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual[1], expected1))

        expected2 = np.array([
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual[2], expected2))

        expected3 = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1],
            [1, 1, 1, 0, 1, 1],
            [1, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual[3], expected3))

        expected4 = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual[4], expected4))

        expected5 = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual[5], expected5))

        expected6 = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual[6], expected6))

    def test_find_objects_with_ignore_mask_inner_20000(self):
        # Arrange
        input_image = np.array([
            [9, 5, 5],
            [5, 9, 5],
            [5, 5, 9],
        ], dtype=np.uint8)
        ignore_mask = np.zeros_like(input_image, dtype=np.uint8)

        # Act
        mask_vec = ConnectedComponent.find_objects_with_ignore_mask_inner(PixelConnectivity.CONNECTIVITY8, input_image, ignore_mask)

        # Assert
        expected = [
            ConnectedComponentItem(np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ], dtype=np.uint8), 3, 0, 0),
            ConnectedComponentItem(np.array([
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0],
            ], dtype=np.uint8), 6, 1, 0)
        ]
        self.assertEqual(mask_vec, expected)

    def test_find_objects_with_ignore_mask_inner_20001(self):
        # Arrange
        input_image = np.array([
            [5, 5, 5, 5],
            [5, 5, 9, 9],
            [9, 5, 5, 5],
            [9, 9, 5, 5],
        ], dtype=np.uint8)
        ignore_mask = np.array([
            [1, 1, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ], dtype=np.uint8)

        # Act
        mask_vec = ConnectedComponent.find_objects_with_ignore_mask_inner(PixelConnectivity.CONNECTIVITY8, input_image, ignore_mask)

        # Assert
        expected = [
            ConnectedComponentItem(np.array([
                [0, 0, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ], dtype=np.uint8), 2, 2, 1),
            ConnectedComponentItem(np.array([
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 1, 0, 0],
            ], dtype=np.uint8), 3, 0, 2)
        ]
        self.assertEqual(mask_vec, expected)

    def test_find_objects_all_30000(self):
        # Arrange
        input_image = np.array([
            [9, 5, 5],
            [5, 9, 5],
            [5, 5, 9],
        ], dtype=np.uint8)

        # Act
        mask_vec = ConnectedComponent.find_objects(PixelConnectivity.CONNECTIVITY8, input_image)

        # Assert
        self.assertEqual(len(mask_vec), 2)
        output = np.vstack(mask_vec)
        expected = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],

            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(output, expected))

    def test_find_objects_with_ignore_mask_40000(self):
        # Arrange
        input_image = np.array([
            [9, 5, 5],
            [5, 9, 5],
            [5, 5, 9],
        ], dtype=np.uint8)
        ignore_mask = np.array([
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ], dtype=np.uint8)

        # Act
        mask_vec = ConnectedComponent.find_objects_with_ignore_mask(PixelConnectivity.CONNECTIVITY8, input_image, ignore_mask)

        # Assert
        self.assertEqual(len(mask_vec), 2)
        output = np.vstack(mask_vec)
        expected = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 0],

            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(output, expected))

    def test_find_objects_with_ignore_mask_40001(self):
        # Arrange
        input_image = np.array([
            [5, 5, 6, 6],
            [5, 5, 6, 6],
            [5, 5, 6, 6],
            [5, 5, 6, 6],
        ], dtype=np.uint8)
        ignore_mask = np.array([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ], dtype=np.uint8)

        # Act
        mask_vec = ConnectedComponent.find_objects_with_ignore_mask(PixelConnectivity.CONNECTIVITY8, input_image, ignore_mask)

        # Assert
        self.assertEqual(len(mask_vec), 2)
        output = np.vstack(mask_vec)
        expected = np.array([
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],

            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(output, expected))

if __name__ == "__main__":
    unittest.main()
