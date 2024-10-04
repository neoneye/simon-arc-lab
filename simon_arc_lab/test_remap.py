import unittest
from .remap import remap

class TestRemap(unittest.TestCase):
    def test_10000_remap(self):
        self.assertAlmostEqual(remap(150, 100, 200, 10, 20), 15, delta=0.0001)
        self.assertAlmostEqual(remap(150, 100, 200, -10, -20), -15, delta=0.0001)
        self.assertAlmostEqual(remap(-100, -100, 100, 1, 5), 1, delta=0.0001)
        self.assertAlmostEqual(remap(-50, -100, 100, 1, 5), 2, delta=0.0001)
        self.assertAlmostEqual(remap(0, -100, 100, 1, 5), 3, delta=0.0001)
        self.assertAlmostEqual(remap(50, -100, 100, 1, 5), 4, delta=0.0001)
        self.assertAlmostEqual(remap(100, -100, 100, 1, 5), 5, delta=0.0001)
