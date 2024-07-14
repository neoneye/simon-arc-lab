import unittest
import numpy as np
from .deserialize import deserialize, decode_rle_row, decode_rle_row_inner

class TestDeserialize(unittest.TestCase):
    def test_decode_rle_row_inner_0(self):
        a = "z6"
        actual = decode_rle_row_inner(a)
        expected = [6] * 27
        self.assertEqual(actual, expected)

    def test_decode_rle_row_inner_1(self):
        a = "z66"
        actual = decode_rle_row_inner(a)
        expected = [6] * 28
        self.assertEqual(actual, expected)

    def test_decode_rle_row_0(self):
        a = "0"
        actual = decode_rle_row(a, 11)
        expected = [0] * 11
        self.assertEqual(actual, expected)

    def test_decode_rle_row_1(self):
        a = "02a12e0"
        actual = decode_rle_row(a, 11)
        expected = [0, 2, 1, 1, 2, 0, 0, 0, 0, 0, 0]
        self.assertEqual(actual, expected)

    def test_decode_rle_row_2(self):
        a = "z98"
        actual = decode_rle_row(a, 28)
        expected = [9] * 27 + [8]
        self.assertEqual(actual, expected)

    def test_decode_rle_row_3(self):
        a = "d7d7"
        actual = decode_rle_row(a, 10)
        expected = [7] * 5 + [7] * 5
        self.assertEqual(actual, expected)

    def test_decode_rle_row_4(self):
        a = "6666"
        actual = decode_rle_row(a, 4)
        expected = [6] * 4
        self.assertEqual(actual, expected)

    def test_deserialize_full(self):
        a = "11 12 0,0c2e0,02a12e0,,,,,,0c2e0,0,,"
        actual = deserialize(a)
        # print(actual)

        # Create a 11x11 array filled with zeros
        expected = np.zeros((12, 11), dtype=np.uint8)

        # Fill in the specific values
        expected[1:9, 1:5] = [
            [2, 2, 2, 2],
            [2, 1, 1, 2],
            [2, 1, 1, 2],
            [2, 1, 1, 2],
            [2, 1, 1, 2],
            [2, 1, 1, 2],
            [2, 1, 1, 2],
            [2, 2, 2, 2]]
        # print(expected)
        self.assertTrue(np.array_equal(actual, expected))

if __name__ == '__main__':
    unittest.main()
