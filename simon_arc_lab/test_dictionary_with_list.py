import unittest
from .dictionary_with_list import DictionaryWithList

class TestDictionaryWithList(unittest.TestCase):
    def test_10000_length_of_lists_none(self):
        actual = DictionaryWithList.length_of_lists(None)
        expected = None
        self.assertEqual(actual, expected)

    def test_10001_length_of_lists_empty_dict(self):
        d = dict()
        actual = DictionaryWithList.length_of_lists(d)
        expected = None
        self.assertEqual(actual, expected)

    def test_10002_length_of_lists_same(self):
        d = {
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        }
        actual = DictionaryWithList.length_of_lists(d)
        expected = (3, 3)
        self.assertEqual(actual, expected)

    def test_10003_length_of_lists_different(self):
        d = {
            'a': [1, 2, 3],
            'b': [4, 5, 6, 7]
        }
        actual = DictionaryWithList.length_of_lists(d)
        expected = (3, 4)
        self.assertEqual(actual, expected)

    def test_20000_merge_two_dictionaries_with_suffix(self):
        dict0 = {
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        }
        dict1 = {
            'a': [7, 8, 9],
            'b': [10, 11, 12]
        }
        actual = DictionaryWithList.merge_two_dictionaries_with_suffix(dict0, dict1)
        expected = {
            'a_0': [1, 2, 3],
            'b_0': [4, 5, 6],
            'a_1': [7, 8, 9],
            'b_1': [10, 11, 12]
        }
        self.assertEqual(actual, expected)
