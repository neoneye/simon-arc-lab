import unittest
from .rectangle import *
from .rectangle_getarea import *

class TestRectangleGetArea(unittest.TestCase):
    def test_10000_getarea_top_40even(self):
        actual = rectangle_getarea(Rectangle(10, 20, 30, 40), 'top')
        self.assertEqual(actual, Rectangle(10, 20, 30, 20))

    def test_10001_getarea_top_41odd(self):
        actual = rectangle_getarea(Rectangle(10, 20, 30, 41), 'top')
        self.assertEqual(actual, Rectangle(10, 20, 30, 21))

    def test_10002_getarea_top_42odd(self):
        actual = rectangle_getarea(Rectangle(10, 20, 30, 42), 'top')
        self.assertEqual(actual, Rectangle(10, 20, 30, 21))

    def test_11000_getarea_bottom_40even(self):
        actual = rectangle_getarea(Rectangle(10, 20, 30, 40), 'bottom')
        self.assertEqual(actual, Rectangle(10, 40, 30, 20))

    def test_11001_getarea_bottom_41odd(self):
        actual = rectangle_getarea(Rectangle(10, 20, 30, 41), 'bottom')
        self.assertEqual(actual, Rectangle(10, 40, 30, 21))

    def test_11002_getarea_bottom_42even(self):
        actual = rectangle_getarea(Rectangle(10, 20, 30, 42), 'bottom')
        self.assertEqual(actual, Rectangle(10, 41, 30, 21))

    def test_12000_getarea_left_40even(self):
        actual = rectangle_getarea(Rectangle(20, 10, 40, 30), 'left')
        self.assertEqual(actual, Rectangle(20, 10, 20, 30))

    def test_12001_getarea_left_41odd(self):
        actual = rectangle_getarea(Rectangle(20, 10, 41, 30), 'left')
        self.assertEqual(actual, Rectangle(20, 10, 21, 30))

    def test_12002_getarea_left_42even(self):
        actual = rectangle_getarea(Rectangle(20, 10, 42, 30), 'left')
        self.assertEqual(actual, Rectangle(20, 10, 21, 30))

    def test_13000_getarea_right_40even(self):
        actual = rectangle_getarea(Rectangle(20, 10, 40, 30), 'right')
        self.assertEqual(actual, Rectangle(40, 10, 20, 30))

    def test_13001_getarea_right_41odd(self):
        actual = rectangle_getarea(Rectangle(20, 10, 41, 30), 'right')
        self.assertEqual(actual, Rectangle(40, 10, 21, 30))

    def test_13002_getarea_right_42even(self):
        actual = rectangle_getarea(Rectangle(20, 10, 42, 30), 'right')
        self.assertEqual(actual, Rectangle(41, 10, 21, 30))

