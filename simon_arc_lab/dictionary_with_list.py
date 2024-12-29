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
