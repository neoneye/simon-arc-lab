import unittest
from .task_color_profile import *

class TestTaskColorProfile(unittest.TestCase):
    def test_compute_color_insert_remove(self):
        # Arrange
        filename = 'testdata/ARC-AGI/data/training/2bcee788.json'
        task = Task.load_arcagi1(filename)
        # Act
        profile = TaskColorProfile(task)
        # Assert
        self.assertEqual(profile.color_insert_intersection, {3})
        self.assertEqual(profile.color_remove_intersection, {0, 2})

    def test_compute_optional_color_insert(self):
        # Arrange
        filename = 'testdata/ARC-AGI/data/training/253bf280.json'
        task = Task.load_arcagi1(filename)
        # Act
        profile = TaskColorProfile(task)
        # Assert
        self.assertEqual(profile.optional_color_insert_set, {3})
        self.assertEqual(profile.has_optional_color_insert, True)

    def test_compute_color_mapping(self):
        # Arrange
        filename = 'testdata/ARC-AGI/data/evaluation/6ea4a07e.json'
        task = Task.load_arcagi1(filename)
        # Act
        profile = TaskColorProfile(task)
        # Assert
        expected = {
            frozenset({8, 0}): {0, 2}, 
            frozenset({0, 3}): {0, 1}, 
            frozenset({0, 5}): {0, 4}
        }
        self.assertEqual(profile.color_mapping, expected)

    def test_compute_number_of_unique_colors_in_outputs(self):
        # Arrange
        filename = 'testdata/ARC-AGI/data/evaluation/009d5c81.json'
        task = Task.load_arcagi1(filename)
        # Act
        profile = TaskColorProfile(task)
        # Assert
        self.assertEqual(profile.number_of_unique_colors_in_outputs, 2)

    def test_same_histogram_for_input_output(self):
        # Arrange
        filename = 'testdata/ARC-AGI/data/training/e9afcf9a.json'
        task = Task.load_arcagi1(filename)
        # Act
        profile = TaskColorProfile(task)
        # Assert
        self.assertEqual(profile.same_histogram_for_input_output, True)

    def test_same_unique_colors_across_all_images(self):
        # Arrange
        filename = 'testdata/ARC-AGI/data/training/3618c87e.json'
        task = Task.load_arcagi1(filename)
        # Act
        profile = TaskColorProfile(task)
        # Assert
        self.assertEqual(profile.same_unique_colors_across_all_images, True)

    def test_same_unique_colors_for_all_outputs(self):
        # Arrange
        filename = 'testdata/ARC-AGI/data/training/db3e9e38.json'
        task = Task.load_arcagi1(filename)
        # Act
        profile = TaskColorProfile(task)
        # Assert
        self.assertEqual(profile.same_unique_colors_for_all_outputs, True)

    def test_remove_duplicates_keep_first0(self):
        input = [
            (True, {5, 6}),
            (False, {5, 6}),
        ]
        actual = TaskColorProfile.remove_duplicates(input)
        expected = [
            (True, {5, 6}),
        ]
        self.assertEqual(actual, expected)

    def test_remove_duplicates_keep_first1(self):
        input = [
            (False, {5, 6}),
            (True, {5, 6}),
        ]
        actual = TaskColorProfile.remove_duplicates(input)
        expected = [
            (False, {5, 6}),
        ]
        self.assertEqual(actual, expected)

    def test_remove_duplicates_multiple_identical(self):
        input = [
            (False, {5, 6}),
            (False, {5, 6}),
            (False, {5, 6}),
            (True, {7, 8, 9}),
            (True, {7, 8, 9}),
        ]
        actual = TaskColorProfile.remove_duplicates(input)
        expected = [
            (False, {5, 6}),
            (True, {7, 8, 9}),
        ]
        self.assertEqual(actual, expected)

    def test_remove_duplicates_keep_first(self):
        input = [
            (True, {5, 6}),
            (True, {3, 9}),
            (False, {1}),
            (True, {3, 9}),
            (True, {5, 6}),
        ]
        actual = TaskColorProfile.remove_duplicates(input)
        expected = [
            (True, {5, 6}),
            (True, {3, 9}),
            (False, {1}),
        ]
        self.assertEqual(actual, expected)

    def test_predict_output_colors_for_test_index_17cae0c1(self):
        # Arrange
        filename = 'testdata/ARC-AGI/data/evaluation/17cae0c1.json'
        task = Task.load_arcagi1(filename)
        profile = TaskColorProfile(task)
        # Act
        actual = profile.predict_output_colors_for_test_index(0)
        # Assert
        expected = [
            (False, {1, 3, 4, 6, 9})
        ]
        self.assertEqual(actual.certain_colorset_list, expected)

    def test_predict_output_colors_for_test_index_009d5c81(self):
        # Arrange
        filename = 'testdata/ARC-AGI/data/evaluation/009d5c81.json'
        task = Task.load_arcagi1(filename)
        profile = TaskColorProfile(task)
        # Act
        actual = profile.predict_output_colors_for_test_index(0)
        # Assert
        expected = [
            (False, {0, 2, 3, 7})
        ]
        self.assertEqual(actual.certain_colorset_list, expected)

