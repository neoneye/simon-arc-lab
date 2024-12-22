import unittest
from .digits_from_response import digits_from_response

RESPONSE_VALID_A = """
text before

Input:

3 2
7 8

Output:

1 2 3
4 5 6

text after
"""

RESPONSE_INVALID_A = """
text before

No digits anywhere

text after
"""

RESPONSE_INVALID_B = """
text before

Some digits in the middle (1 2 3) of the line

text after
"""

class TestDigitsFromResponse(unittest.TestCase):
    def test_10000_response_valid_a(self):
        actual = digits_from_response(RESPONSE_VALID_A)
        expected = [
            [1, 2, 3], 
            [4, 5, 6]
        ]
        self.assertEqual(actual, expected)

    def test_20000_response_invalid_a(self):
        with self.assertRaises(ValueError):
            digits_from_response(RESPONSE_INVALID_A)


    def test_20001_response_invalid_a(self):
        with self.assertRaises(ValueError):
            digits_from_response(RESPONSE_INVALID_B)

