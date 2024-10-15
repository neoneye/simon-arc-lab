import unittest
import numpy as np
from .image_vote import *

class TestImageVote(unittest.TestCase):
    def test_vote_unambiguous(self):
        image0 = np.array([
            [1, 1, 5], 
            [1, 1, 1], 
            [2, 2, 2],
            [2, 2, 2]], dtype=np.uint8)
        image1 = np.array([
            [3, 1, 1], 
            [1, 1, 1], 
            [2, 5, 2],
            [2, 2, 2]], dtype=np.uint8)
        image2 = np.array([
            [1, 1, 1], 
            [1, 1, 2], 
            [2, 2, 1],
            [2, 2, 2]], dtype=np.uint8)
        images = [image0, image1, image2]
        actual = image_vote(images)
        expected = np.array([
            [1, 1, 1], 
            [1, 1, 1], 
            [2, 2, 2],
            [2, 2, 2]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_vote_ambiguous_sizes(self):
        image0 = np.array([
            [1, 1, 5], 
            [2, 2, 2],
            [2, 2, 2]], dtype=np.uint8)
        image1 = np.array([
            [1, 1, 5], 
            [1, 1, 1], 
            [2, 2, 2],
            [2, 2, 2]], dtype=np.uint8)
        image2 = np.array([
            [3, 1, 1], 
            [1, 1, 1], 
            [2, 5, 2],
            [2, 2, 2]], dtype=np.uint8)
        image3 = np.array([
            [1, 1, 1], 
            [1, 1, 2], 
            [2, 2, 1],
            [2, 2, 2]], dtype=np.uint8)
        images = [image0, image1, image2, image3]
        actual = image_vote(images)
        expected = np.array([
            [1, 1, 1], 
            [1, 1, 1], 
            [2, 2, 2],
            [2, 2, 2]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))
