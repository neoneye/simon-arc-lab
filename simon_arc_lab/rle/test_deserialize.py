import unittest
import numpy as np
from .deserialize import deserialize, decode_rle_row, decode_rle_row_inner, DecodeRLEError, DeserializeError

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

    def test_deserialize_error_3parts(self):
        junk = "1 2 3 4"
        with self.assertRaises(DeserializeError) as context:
            deserialize(junk)
        self.assertTrue("Expected 3 parts" in str(context.exception))
        self.assertAlmostEqual(0.0, context.exception.score, delta=0.0001)

    def test_deserialize_error_width_junk(self):
        junk = "x 12 0,0c2e0,02a12e0,,,,,,0c2e0,0,,"
        with self.assertRaises(DeserializeError) as context:
            deserialize(junk)
        self.assertTrue("Cannot parse width" in str(context.exception))
        self.assertAlmostEqual(1.0, context.exception.score, delta=0.0001)

    def test_deserialize_error_width_negative(self):
        junk = "-11 12 0,0c2e0,02a12e0,,,,,,0c2e0,0,,"
        with self.assertRaises(DeserializeError) as context:
            deserialize(junk)
        self.assertTrue("Width must 1 or greater" in str(context.exception))
        self.assertAlmostEqual(2.0, context.exception.score, delta=0.0001)

    def test_deserialize_error_width_zero(self):
        junk = "0 12 0,0c2e0,02a12e0,,,,,,0c2e0,0,,"
        with self.assertRaises(DeserializeError) as context:
            deserialize(junk)
        self.assertTrue("Width must 1 or greater" in str(context.exception))
        self.assertAlmostEqual(2.0, context.exception.score, delta=0.0001)

    def test_deserialize_error_width_exceed_max(self):
        junk = "100 1 z0z0z0z0"
        with self.assertRaises(DeserializeError) as context:
            deserialize(junk)
        self.assertTrue("Width exceeds MAX_IMAGE_SIZE" in str(context.exception))
        self.assertEqual("width: 100, MAX_IMAGE_SIZE: 99", context.exception.details)
        self.assertAlmostEqual(3.0, context.exception.score, delta=0.0001)

    def test_deserialize_error_height_junk(self):
        junk = "11 x 0,0c2e0,02a12e0,,,,,,0c2e0,0,,"
        with self.assertRaises(DeserializeError) as context:
            deserialize(junk)
        self.assertTrue("Cannot parse height" in str(context.exception))
        self.assertAlmostEqual(4.0, context.exception.score, delta=0.0001)

    def test_deserialize_error_height_negative(self):
        junk = "11 -12 0,0c2e0,02a12e0,,,,,,0c2e0,0,,"
        with self.assertRaises(DeserializeError) as context:
            deserialize(junk)
        self.assertTrue("Height must 1 or greater" in str(context.exception))
        self.assertAlmostEqual(5.0, context.exception.score, delta=0.0001)

    def test_deserialize_error_height_zero(self):
        junk = "11 0 0,0c2e0,02a12e0,,,,,,0c2e0,0,,"
        with self.assertRaises(DeserializeError) as context:
            deserialize(junk)
        self.assertTrue("Height must 1 or greater" in str(context.exception))
        self.assertAlmostEqual(5.0, context.exception.score, delta=0.0001)

    def test_deserialize_error_height_exceed_max(self):
        junk = "1 100 7,,,,,,,,,"
        with self.assertRaises(DeserializeError) as context:
            deserialize(junk)
        self.assertTrue("Height exceeds MAX_IMAGE_SIZE" in str(context.exception))
        self.assertEqual("height: 100, MAX_IMAGE_SIZE: 99", context.exception.details)
        self.assertAlmostEqual(6.0, context.exception.score, delta=0.0001)

    def test_deserialize_error_heightmismatch_way_too_few(self):
        junk = "1 15 0,,,,,,,,,,,"
        with self.assertRaises(DeserializeError) as context:
            deserialize(junk)
        self.assertTrue("Too few rows. Mismatch between height and the number of RLE rows" in str(context.exception))
        self.assertEqual("height: 15 != count_rows: 12", context.exception.details)
        self.assertAlmostEqual(7.0, context.exception.score, delta=0.0001)

    def test_deserialize_error_heightmismatch_way_too_many(self):
        junk = "1 5 0,,,,,,,,,,,"
        with self.assertRaises(DeserializeError) as context:
            deserialize(junk)
        self.assertTrue("Too many rows. Mismatch between height and the number of RLE rows" in str(context.exception))
        self.assertEqual("height: 5 != count_rows: 12", context.exception.details)
        self.assertAlmostEqual(8.0, context.exception.score, delta=0.0001)

    def test_deserialize_error_heightmismatch_one_too_few(self):
        junk = "1 7 0,,,,,"
        with self.assertRaises(DeserializeError) as context:
            deserialize(junk)
        self.assertTrue("Too few rows, only 1 row missing. Mismatch between height and the number of RLE rows" in str(context.exception))
        self.assertEqual("height: 7 != count_rows: 6", context.exception.details)
        self.assertAlmostEqual(9.0, context.exception.score, delta=0.0001)

    def test_deserialize_error_heightmismatch_one_too_many(self):
        junk = "1 5 0,,,,,"
        with self.assertRaises(DeserializeError) as context:
            deserialize(junk)
        self.assertTrue("Too many rows, only 1 row too many. Mismatch between height and the number of RLE rows" in str(context.exception))
        self.assertEqual("height: 5 != count_rows: 6", context.exception.details)
        self.assertAlmostEqual(10.0, context.exception.score, delta=0.0001)

    def test_deserialize_error_emptyfirstrow(self):
        junk = "1 2 ,5"
        with self.assertRaises(DeserializeError) as context:
            deserialize(junk)
        self.assertTrue("First row is empty" in str(context.exception))
        self.assertAlmostEqual(11.0, context.exception.score, delta=0.0001)

    def test_deserialize_error_invalidcharacterforfullrow(self):
        junk = "1 2 3,#"
        with self.assertRaises(DeserializeError) as context:
            deserialize(junk)
        self.assertTrue("Cannot deserialize row" in str(context.exception))
        self.assertEqual("y: 1 height: 2", context.exception.details)
        self.assertAlmostEqual(99.0, context.exception.score, delta=0.0001)
        decode_rle_error = context.exception.decode_rle_error
        self.assertIsInstance(decode_rle_error, DecodeRLEError)
        self.assertTrue("Invalid character for full row" in str(decode_rle_error))

    def test_deserialize_error_invalidcharacterinsiderow(self):
        junk = "1 2 {3,"
        with self.assertRaises(DeserializeError) as context:
            deserialize(junk)
        self.assertTrue("Cannot deserialize row" in str(context.exception))
        self.assertEqual("y: 0 height: 2", context.exception.details)
        self.assertAlmostEqual(12.0, context.exception.score, delta=0.0001)
        decode_rle_error = context.exception.decode_rle_error
        self.assertIsInstance(decode_rle_error, DecodeRLEError)
        self.assertTrue("Invalid character inside row" in str(decode_rle_error))
        self.assertEqual("Character: {", decode_rle_error.details)

    def test_deserialize_error_adjacent_az(self):
        junk = "1 2 aa3,"
        with self.assertRaises(DeserializeError) as context:
            deserialize(junk)
        self.assertTrue("Cannot deserialize row" in str(context.exception))
        self.assertEqual("y: 0 height: 2", context.exception.details)
        self.assertAlmostEqual(12.0, context.exception.score, delta=0.0001)
        decode_rle_error = context.exception.decode_rle_error
        self.assertIsInstance(decode_rle_error, DecodeRLEError)
        self.assertTrue("No adjacent a-z characters are allowed" in str(decode_rle_error))
        self.assertEqual("Character: a", decode_rle_error.details)

    def test_deserialize_error_last_character_1row(self):
        junk = "3 1 345a"
        with self.assertRaises(DeserializeError) as context:
            deserialize(junk)
        self.assertTrue("Cannot deserialize row" in str(context.exception))
        self.assertEqual("y: 0 height: 1", context.exception.details)
        self.assertAlmostEqual(12.0, context.exception.score, delta=0.0001)
        decode_rle_error = context.exception.decode_rle_error
        self.assertIsInstance(decode_rle_error, DecodeRLEError)
        self.assertTrue("Last character must not be a-z character" in str(decode_rle_error))
        self.assertEqual("Character: a", decode_rle_error.details)

    def test_deserialize_error_last_character_2rows(self):
        junk = "3 2 678,345a"
        with self.assertRaises(DeserializeError) as context:
            deserialize(junk)
        self.assertTrue("Cannot deserialize row" in str(context.exception))
        self.assertEqual("y: 1 height: 2", context.exception.details)
        self.assertAlmostEqual(99.0, context.exception.score, delta=0.0001)
        decode_rle_error = context.exception.decode_rle_error
        self.assertIsInstance(decode_rle_error, DecodeRLEError)
        self.assertTrue("Last character must not be a-z character" in str(decode_rle_error))
        self.assertEqual("Character: a", decode_rle_error.details)

    def test_deserialize_error_mismatch_width_medium_score(self):
        junk = "4 5 7878,9797,78785,9797,1234"
        with self.assertRaises(DeserializeError) as context:
            deserialize(junk)
        self.assertTrue("Cannot deserialize row" in str(context.exception))
        self.assertEqual("y: 2 height: 5", context.exception.details)
        self.assertAlmostEqual(55.5, context.exception.score, delta=0.0001)
        decode_rle_error = context.exception.decode_rle_error
        self.assertIsInstance(decode_rle_error, DecodeRLEError)
        self.assertTrue("Mismatch between width and the number of RLE columns" in str(decode_rle_error))
        self.assertEqual("Expected width: 4, Decoded width: 5", decode_rle_error.details)

    def test_deserialize_error_mismatch_width(self):
        junk = "4 5 7878,9797,7878,9797,12345"
        with self.assertRaises(DeserializeError) as context:
            deserialize(junk)
        self.assertTrue("Cannot deserialize row" in str(context.exception))
        self.assertEqual("y: 4 height: 5", context.exception.details)
        self.assertAlmostEqual(99.0, context.exception.score, delta=0.0001)
        decode_rle_error = context.exception.decode_rle_error
        self.assertIsInstance(decode_rle_error, DecodeRLEError)
        self.assertTrue("Mismatch between width and the number of RLE columns" in str(decode_rle_error))
        self.assertEqual("Expected width: 4, Decoded width: 5", decode_rle_error.details)

if __name__ == '__main__':
    unittest.main()
