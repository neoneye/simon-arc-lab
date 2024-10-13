import unittest
import numpy as np
from .image_analyze import *

class TestImageAnalyze(unittest.TestCase):
    def test_10000_union(self):
        # Arrange
        a = AnalyzeBase()
        a.set_feature_to_valueset(Feature(FeatureType.TEST_A), set('x'))
        b = AnalyzeBase()
        b.set_feature_to_valueset(Feature(FeatureType.TEST_A), set('y'))
        # Act
        intersection, union = AnalyzeBase.intersection_union_emptyset([a, b], Feature(FeatureType.TEST_A))
        # Assert
        self.assertEqual(union, set(['x', 'y']))

    def test_20000_intersection_empty(self):
        # Arrange
        a = AnalyzeBase()
        a.set_feature_to_valueset(Feature(FeatureType.TEST_A), set('x'))
        b = AnalyzeBase()
        # Act
        intersection, union = AnalyzeBase.intersection_union_emptyset([a, b], Feature(FeatureType.TEST_A))
        # Assert
        self.assertEqual(intersection, set())

    def test_20001_intersection_empty(self):
        # Arrange
        a = AnalyzeBase()
        a.set_feature_to_valueset(Feature(FeatureType.TEST_A), set('x'))
        b = AnalyzeBase()
        a.set_feature_to_valueset(Feature(FeatureType.TEST_A), set('y'))
        # Act
        intersection, union = AnalyzeBase.intersection_union_emptyset([a, b], Feature(FeatureType.TEST_A))
        # Assert
        self.assertEqual(intersection, set())

    def test_20002_intersection_empty(self):
        # Arrange
        a = AnalyzeBase()
        a.set_feature_to_valueset(Feature(FeatureType.TEST_A), set('x'))
        b = AnalyzeBase()
        a.set_feature_to_valueset(Feature(FeatureType.TEST_A), set(['y', 'z']))
        # Act
        intersection, union = AnalyzeBase.intersection_union_emptyset([a, b], Feature(FeatureType.TEST_A))
        # Assert
        self.assertEqual(intersection, set())

    def test_20100_intersection_nonempty(self):
        # Arrange
        a = AnalyzeBase()
        a.set_feature_to_valueset(Feature(FeatureType.TEST_A), set('x'))
        b = AnalyzeBase()
        b.set_feature_to_valueset(Feature(FeatureType.TEST_A), set('x'))
        # Act
        intersection, union = AnalyzeBase.intersection_union_emptyset([a, b], Feature(FeatureType.TEST_A))
        # Assert
        self.assertEqual(intersection, set('x'))

    def test_20101_intersection_nonempty(self):
        # Arrange
        a = AnalyzeBase()
        a.set_feature_to_valueset(Feature(FeatureType.TEST_A), set(['x', 'y']))
        b = AnalyzeBase()
        b.set_feature_to_valueset(Feature(FeatureType.TEST_A), set(['y', 'z']))
        # Act
        intersection, union = AnalyzeBase.intersection_union_emptyset([a, b], Feature(FeatureType.TEST_A))
        # Assert
        self.assertEqual(intersection, set('y'))

    def test_20102_intersection_nonempty(self):
        # Arrange
        a = AnalyzeBase()
        a.set_feature_to_valueset(Feature(FeatureType.TEST_A), set(['x', 'y', 'z']))
        b = AnalyzeBase()
        b.set_feature_to_valueset(Feature(FeatureType.TEST_A), set(['y', 'z', 'w']))
        # Act
        intersection, union = AnalyzeBase.intersection_union_emptyset([a, b], Feature(FeatureType.TEST_A))
        # Assert
        self.assertEqual(intersection, set(['y', 'z']))

    def test_20200_counter_set_of_strings(self):
        # Arrange
        a = AnalyzeBase()
        a.set_feature_to_valueset(Feature(FeatureType.TEST_A), set(['x', 'y', 'z']))
        b = AnalyzeBase()
        b.set_feature_to_valueset(Feature(FeatureType.TEST_A), set(['y', 'z', 'w']))
        c = AnalyzeBase()
        c.set_feature_to_valueset(Feature(FeatureType.TEST_A), set(['x', 'y', 'z']))
        # Act
        counter = AnalyzeBase.counter_emptyset([a, b, c], Feature(FeatureType.TEST_A))
        # Assert
        self.assertEqual(counter["['x', 'y', 'z']"], 2)

    def test_20201_counter_set_of_ints(self):
        # Arrange
        a = AnalyzeBase()
        a.set_feature_to_valueset(Feature(FeatureType.TEST_A), {42})
        b = AnalyzeBase()
        b.set_feature_to_valueset(Feature(FeatureType.TEST_A), {42})
        c = AnalyzeBase()
        c.set_feature_to_valueset(Feature(FeatureType.TEST_A), {42})
        d = AnalyzeBase()
        d.set_feature_to_valueset(Feature(FeatureType.TEST_A), {42, 5})
        # Act
        counter = AnalyzeBase.counter_emptyset([a, b, c, d], Feature(FeatureType.TEST_A))
        # Assert
        self.assertEqual(counter["[42]"], 3)

    def test_30000_image_analyze_number_of_unique_colors_leftright(self):
        # Arrange
        image = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        ia = ImageAnalyze(image)
        analyze_line_list = ia.analyze_line_for_leftright()
        # Act
        intersection, union = AnalyzeBase.intersection_union_emptyset(analyze_line_list, Feature(FeatureType.NUMBER_OF_UNIQUE_COLORS))
        # Assert
        self.assertEqual(intersection, {2})
        self.assertEqual(union, {2})

    def test_30001_image_analyze_number_of_unique_colors_topbottom(self):
        # Arrange
        image = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        ia = ImageAnalyze(image)
        analyze_line_list = ia.analyze_line_for_topbottom()
        # Act
        intersection, union = AnalyzeBase.intersection_union_emptyset(analyze_line_list, Feature(FeatureType.NUMBER_OF_UNIQUE_COLORS))
        # Assert
        self.assertEqual(intersection, {3})
        self.assertEqual(union, {3})

    def test_40000_image_analyze_unique_colors(self):
        # Arrange
        image = np.array([
            [1, 4, 1, 5],
            [4, 2, 5, 3],
            [3, 4, 5, 1]], dtype=np.uint8)
        ia = ImageAnalyze(image)
        analyze_line_list = ia.analyze_line_for_leftright()
        # Act
        intersection, union = AnalyzeBase.intersection_union_emptyset(analyze_line_list, Feature(FeatureType.UNIQUE_COLORS))
        # Assert
        self.assertEqual(intersection, {4, 5})
        self.assertEqual(union, {1, 2, 3, 4, 5})

    def test_50000_image_analyze_compressed_representation_with_intersection(self):
        # Arrange
        image = np.array([
            [1, 2, 2, 3],
            [1, 1, 2, 3],
            [1, 2, 3, 3]], dtype=np.uint8)
        ia = ImageAnalyze(image)
        analyze_line_list = ia.analyze_line_for_leftright()
        # Act
        intersection, union = AnalyzeBase.intersection_union_emptyset(analyze_line_list, Feature(FeatureType.COMPRESSED_REPRESENTATION))
        # Assert
        self.assertEqual(intersection, {(1, 2, 3)})
        self.assertEqual(union, {(1, 2, 3)})

    def test_50001_image_analyze_compressed_representation_without_intersection(self):
        # Arrange
        image = np.array([
            [3, 2, 1, 1],
            [1, 2, 2, 3],
            [1, 1, 2, 3],
            [1, 2, 3, 3]], dtype=np.uint8)
        ia = ImageAnalyze(image)
        analyze_line_list = ia.analyze_line_for_leftright()
        # Act
        intersection, union = AnalyzeBase.intersection_union_emptyset(analyze_line_list, Feature(FeatureType.COMPRESSED_REPRESENTATION))
        # Assert
        self.assertEqual(intersection, set())
        self.assertEqual(union, {(1, 2, 3), (3, 2, 1)})

    def test_50002_image_analyze_compressed_representation_without_intersection(self):
        # Arrange
        image = np.array([
            [3, 3, 2, 2],
            [7, 7, 7, 7],
            [3, 2, 1, 1],
            [1, 2, 2, 3],
            [1, 1, 2, 3],
            [1, 2, 3, 3]], dtype=np.uint8)
        ia = ImageAnalyze(image)
        analyze_line_list = ia.analyze_line_for_leftright()
        # Act
        intersection, union = AnalyzeBase.intersection_union_emptyset(analyze_line_list, Feature(FeatureType.COMPRESSED_REPRESENTATION))
        # Assert
        self.assertEqual(intersection, set())
        self.assertEqual(union, {(1, 2, 3), (3, 2, 1), (3, 2), (7,)})

    def test_60000_image_analyze_number_of_color_transitions(self):
        # Arrange
        image = np.array([
            [1, 2, 2, 3],
            [1, 1, 2, 3],
            [1, 2, 3, 3],
            [7, 8, 8, 9]], dtype=np.uint8)
        ia = ImageAnalyze(image)
        analyze_line_list = ia.analyze_line_for_leftright()
        # Act
        intersection, union = AnalyzeBase.intersection_union_emptyset(analyze_line_list, Feature(FeatureType.LENGTH_OF_COMPRESSED_REPRESENTATION))
        # Assert
        self.assertEqual(intersection, {3})
        self.assertEqual(union, {3})

    def test_70000_image_analyze_analyze(self):
        # Arrange
        image = np.array([
            [1, 2, 2, 3],
            [1, 1, 2, 3],
            [1, 2, 3, 3],
            [7, 8, 8, 9]], dtype=np.uint8)
        ia = ImageAnalyze(image)
        # Act
        actual = ia.analyze()
        # Assert
        self.assertTrue(actual.__contains__('number_of_unique_colors'))
        self.assertTrue(actual.__contains__('length_of_compressed_representation'))

