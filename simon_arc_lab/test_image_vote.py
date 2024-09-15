import unittest
import numpy as np

def image_vote(images: list[np.array]) -> np.array:
    image0 = images[0]
    height, width = image0.shape
    vote = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            counts = {}
            for image in images:
                pixel = image[y, x]
                if pixel in counts:
                    counts[pixel] += 1
                else:
                    counts[pixel] = 1
            vote[y, x] = max(counts, key=counts.get)
    return vote

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
