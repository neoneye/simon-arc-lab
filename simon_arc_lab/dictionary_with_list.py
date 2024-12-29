from typing import Tuple, Optional

class DictionaryWithList:
    @staticmethod
    def length_of_lists(the_dict: dict[list]) -> Optional[Tuple[int, int]]:
        """
        When checking that all the lists in a dictionary have the same length, this function returns the minimum and maximum lengths of the lists.
        If the min and max differ, then the lists have different lengths.
        """
        if not the_dict:
            return None
        lengths = [len(value) for value in the_dict.values()]
        return min(lengths), max(lengths)

    @staticmethod
    def merge_two_dictionaries_with_suffix(dict0: dict, dict1: dict) -> dict:
        """
        Merge two dictionaries with lists as values. The keys must be the same.
        """

        # the keys must be the same
        assert dict0.keys() == dict1.keys()

        # the number of values must be the same
        count_min0, count_max0 = DictionaryWithList.length_of_lists(dict0)
        if count_min0 != count_max0:
            raise ValueError(f'Expected same length of lists of dict0, {count_min0} != {count_max0}')

        count_min1, count_max1 = DictionaryWithList.length_of_lists(dict1)
        if count_min1 != count_max1:
            raise ValueError(f'Expected same length of lists of dict1, {count_min1} != {count_max1}')

        # both dict0 and dict1 should have the length of their lists
        assert count_min0 == count_min1

        result_dict = {}
        # Use different suffixes
        for key in dict0.keys():
            result_dict[key + "_0"] = dict0[key]
            result_dict[key + "_1"] = dict1[key]
        return result_dict
