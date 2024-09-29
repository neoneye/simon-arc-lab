import unittest
import numpy as np
from .task_similarity import *
from .image_similarity import *

class TestTaskSimilarity(unittest.TestCase):
    def test_10000_same_histogram(self):
        # Arrange
        filename = 'testdata/simple_arc_tasks/25ff71a9.json'
        task = Task.load_arcagi1(filename)
        # Act
        ts = TaskSimilarity.create_with_task(task)
        # Assert
        actual_features = ts.example_pair_feature_set_intersection 
        expected_feature = Feature(FeatureType.SAME_HISTOGRAM)
        self.assertTrue(expected_feature in actual_features)

    def test_20000_same_most_popular_color_list(self):
        # Arrange
        filename = 'testdata/simple_arc_tasks/5582e5ca.json'
        task = Task.load_arcagi1(filename)
        # Act
        ts = TaskSimilarity.create_with_task(task)
        # Assert
        actual_features = ts.example_pair_feature_set_intersection 
        expected_feature = Feature(FeatureType.SAME_MOST_POPULAR_COLOR_LIST)
        self.assertTrue(expected_feature in actual_features)
