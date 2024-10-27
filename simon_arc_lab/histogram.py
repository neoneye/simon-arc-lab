from typing import Dict, Tuple, Optional
import random
import numpy as np

class Histogram:
    def __init__(self, color_count: Dict[int, int]):
        self.color_count = color_count
        self.purge_mutable()

    def clone(self) -> 'Histogram':
        """
        Create a copy of the current Histogram instance.
        """
        return Histogram(self.color_count.copy())
    
    @classmethod
    def empty(cls) -> 'Histogram':
        return cls({})

    @classmethod
    def create_with_image(cls, image: np.array) -> 'Histogram':
        """
        Populate a histogram with the colors from an image.
        """
        hist = {}
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                color = image[y, x]
                if color in hist:
                    hist[color] += 1
                else:
                    hist[color] = 1
        return cls(hist)

    @classmethod
    def create_with_image_list(cls, image_list: list[np.array]) -> 'Histogram':
        """
        Populate a histogram with the colors from multiple images.
        """
        histogram = Histogram.empty()
        for image in image_list:
            histogram = histogram.add(cls.create_with_image(image))
        return histogram

    @classmethod
    def create_with_color_set(cls, color_set: set[int]) -> 'Histogram':
        """
        Populate a histogram with the colors from a set.
        If the color is in the set, then the count is 1.
        """
        hist = {}
        for color in color_set:
            hist[color] = 1
        return cls(hist)

    @classmethod
    def create_random(cls, seed: int, min_colors: int, max_colors: int, min_count: int, max_count: int) -> 'Histogram':
        if min_colors > max_colors:
            raise Exception("min_colors must be less than or equal to max_colors")
        if min_count > max_count:
            raise Exception("min_count must be less than or equal to max_count")
        if min_colors < 0:
            raise Exception("min_colors must be greater than or equal to 0")
        if max_colors > 9:
            raise Exception("max_colors must be less than or equal to 9")
        if min_count < 1:
            raise Exception("min_count must be greater than or equal to 1")
        if max_count > 10000:
            raise Exception("max_count must be less than or equal to 10000")
        
        colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        random.Random(seed).shuffle(colors)
        hist = {}
        number_of_colors = random.Random(seed + 1).randint(min_colors, max_colors)
        for color_index in range(number_of_colors):
            count = random.Random(seed + color_index + 2).randint(min_count, max_count)
            hist[colors[color_index]] = count
        return cls(hist)

    def __eq__(self, other):
        if isinstance(other, Histogram):
            self.purge_mutable()
            other.purge_mutable()
            return self.color_count == other.color_count
        return False

    def sorted_color_count_list(self) -> list[Tuple[int, int]]:
        """
        sort by popularity, if there is a tie, sort by color
        """

        items = sorted(self.color_count.items(), key=lambda item: (-item[1], item[0]))
        return items

    def sorted_count_list(self) -> list[int]:
        """
        sort by popularity. Leave out counters that are zero or negative.
        Descending. The biggest counter is first. The smallest counter is last.
        """

        items = sorted(self.color_count.values(), reverse=True)
        return items

    def pretty(self) -> str:
        """
        Comma separated list of unique colors in the histogram. Ordered by popularity.
        """
        histogram = self.clone()
        histogram.purge_mutable()
        color_count_list = histogram.sorted_color_count_list()
        if len(color_count_list) == 0:
            return 'empty'
        return ','.join([f'{color}:{count}' for color, count in color_count_list])

    def increment(self, color: int):
        if color in self.color_count:
            self.color_count[color] += 1
        else:
            self.color_count[color] = 1
        self.purge_mutable()

    def add(self, other: 'Histogram') -> 'Histogram':
        result = self.clone()
        for color, count in other.color_count.items():
            if color in result.color_count:
                result.color_count[color] += count
            else:
                result.color_count[color] = count
        result.purge_mutable()
        return result

    def subtract_and_discard(self, other: 'Histogram') -> 'Histogram':
        result = self.clone()
        for color, count in other.color_count.items():
            if color in result.color_count:
                result.color_count[color] = max(0, result.color_count[color] - count)
        result.purge_mutable()
        return result

    def max(self, other: 'Histogram') -> 'Histogram':
        """
        Find the maximum count of each color in both histograms.
        if a color is not in both histograms, then it get included in the result.

        :param other: the other histogram to compare with
        :return: a new histogram with the maximum counters
        """
        result = self.clone()
        for color, count in other.color_count.items():
            if color in result.color_count:
                result.color_count[color] = max(result.color_count[color], count)
            else:
                result.color_count[color] = count
        result.purge_mutable()
        return result

    def color_intersection_set(self, other: 'Histogram') -> set:
        """
        Find the set of colors that are in both histograms. The overlap.
        """
        return set(self.color_count.keys()) & set(other.color_count.keys())

    def color_intersection_list(self, other: 'Histogram') -> list[int]:
        """
        Sorted list of colors that are in both histograms. The overlap.
        """
        color_set = self.color_intersection_set(other)
        colors = list(color_set)
        colors.sort()
        return colors

    def color_intersection_pretty(self, other: 'Histogram') -> str:
        """
        Comma separated list of unique colors in the both histograms. The overlap.
        """
        colors = self.color_intersection_list(other)
        if len(colors) == 0:
            return 'empty'
        return ','.join([str(color) for color in colors])

    def min(self, other: 'Histogram') -> 'Histogram':
        """
        Find the minimum count of each color in both histograms.
        if a color is not in both histograms, then it doesn't get included in the result.

        :param other: the other histogram to compare with
        :return: a new histogram with the minimum counters
        """
        colors = self.color_intersection_set(other)
        result = Histogram.empty()
        for color in colors:
            if color in self.color_count and color in other.color_count:
                result.color_count[color] = min(self.color_count[color], other.color_count[color])
            else:
                raise Exception("Unreachable code reached")
        result.purge_mutable()
        return result
    
    def number_of_unique_colors(self) -> int:
        """
        Number of unique colors in the histogram.
        """
        histogram = self.clone()
        histogram.purge_mutable()
        return len(histogram.color_count)

    def unique_colors(self) -> list[int]:
        """
        Sorted list of unique colors in the histogram.
        """
        histogram = self.clone()
        histogram.purge_mutable()
        colors = list(histogram.color_count.keys())
        colors.sort()
        return colors

    def unique_colors_set(self) -> set[int]:
        """
        Set of unique colors in the histogram.
        """
        histogram = self.clone()
        histogram.purge_mutable()
        return set(histogram.color_count.keys())

    def unique_colors_pretty(self) -> str:
        """
        Comma separated list of unique colors in the histogram.
        """
        colors = self.unique_colors()
        if len(colors) == 0:
            return 'empty'
        return ','.join([str(color) for color in colors])

    def remove_other_colors(self, other: 'Histogram') -> 'Histogram':
        """
        Remove colors that are in the other histogram.
        Keep the histogram data for colors that are only present in the current histogram.
        If the other histogram has a color that is not in the current histogram, then ignore it.

        :param other: the other histogram to compare with
        :return: a new histogram
        """
        colors = self.color_intersection_set(other)
        result = self.clone()
        for color in colors:
            del result.color_count[color]
        result.purge_mutable()
        return result

    def sum_of_counters(self) -> int:
        """
        Traverse all the colors, add up their amount.
        """
        return sum(self.color_count.values())

    def most_popular_color(self) -> Optional[int]:
        """
        Find the color with the unambiguous highest counter.
        If there is a tie, return None.
        If there are no colors, return None.
        Only consider counters that are 1 or greater. Ignore colors with zero counters or negative counters.
        """
        found_count = 0
        found_color = None
        same_count = 0
        for color, count in self.color_count.items():
            if count > found_count:
                found_count = count
                found_color = color
                same_count = 1
            elif count == found_count:
                same_count += 1
        if same_count > 1:
            return None
        return found_color

    def most_popular_color_list(self) -> list[int]:
        """
        Find the colors with the highest counters.
        If there is one unambiguous most popular color, return that color.
        If there is a tie, return all multiple colors.
        If there are no colors, return an empty list.
        Only consider counters that are 1 or greater. Ignore colors with zero counters or negative counters.
        """
        found_count = 0
        found_colors = []
        for color, count in self.color_count.items():
            if count < 1:
                continue
            if count > found_count:
                found_count = count
                found_colors = [color]
            elif count == found_count:
                found_colors.append(color)
        sorted_colors = sorted(found_colors)
        return sorted_colors

    def least_popular_color(self) -> Optional[int]:
        """
        Find the color with the unambiguous lowest counter.
        If there is a tie, return None.
        If there are no colors, return None.
        Only consider counters that are 1 or greater. Ignore colors with zero counters or negative counters.
        """
        found_count = self.sum_of_counters() + 1
        found_color = None
        same_count = 0
        for color, count in self.color_count.items():
            if count < 1:
                continue
            if count < found_count:
                found_count = count
                found_color = color
                same_count = 1
            elif count == found_count:
                same_count += 1
        if same_count > 1:
            return None
        return found_color

    def least_popular_color_list(self) -> list[int]:
        """
        Find the colors with the lowest counter.
        If there is an unambiguous color, return return a list with just that color.
        If there is a tie, return a list with multiple colors.
        If there are no colors, return an empty list.
        Only consider counters that are 1 or greater. Ignore colors with zero counters or negative counters.
        """
        found_count = self.sum_of_counters() + 1
        found_colors = []
        same_count = 0
        for color, count in self.color_count.items():
            if count < 1:
                continue
            if count < found_count:
                found_count = count
                found_colors = [color]
                same_count = 1
            elif count == found_count:
                same_count += 1
                found_colors.append(color)
        sorted_colors = sorted(found_colors)
        return sorted_colors

    def histogram_without_mostleast_popular_colors(self) -> 'Histogram':
        """
        Ignore the most popular and the least popular colors.
        Return a new histogram with the remaining colors.
        """
        histogram = self.clone()
        for color in self.most_popular_color_list():
            histogram.remove_color(color)
        for color in self.least_popular_color_list():
            histogram.remove_color(color)
        return histogram

    def get_count_for_color(self, color: int) -> int:
        """
        Get the count for a specific color.
        
        If the color is not in the histogram, return 0.
        """
        return self.color_count.get(color, 0)

    def purge_mutable(self):
        """
        Remove colors where the count is less than 1.
        It doesn't make sense having zero counters in the histogram.
        It doesn't make sense having negative counters in the histogram.
        """
        for color in list(self.color_count.keys()):
            if self.color_count[color] < 1:
                del self.color_count[color]
    
    def available_colors(self) -> list[int]:
        """
        Find the color indexes that are not in the histogram.

        If all colors are used, return an empty list.

        The colors are sorted in ascending order.
        """
        self.purge_mutable()
        colors = set(self.color_count.keys())
        available_colors = []
        for color in range(10):
            if color not in colors:
                available_colors.append(color)
        return available_colors

    def first_available_color(self) -> Optional[int]:
        """
        Find the lowest color index that is not in the histogram.

        If all colors are used, return None.
        """
        colors = self.available_colors()
        if len(colors) == 0:
            return None
        return colors[0]

    def remove_color(self, color: int):
        """
        Remove a color from the histogram.
        """
        self.color_count[color] = 0
        self.purge_mutable()

    @classmethod
    def union_intersection(cls, histogram_list: list['Histogram']) -> Tuple[set, set]:
        """
        Find the union and intersection of multiple histograms.
        """
        union = set()
        intersection = set()
        for index, histogram in enumerate(histogram_list):
            color_set = histogram.unique_colors_set()
            union = union | color_set
            if index == 0:
                intersection = color_set
            else:
                intersection = intersection & color_set
        return (union, intersection)
