from typing import Dict, List, Tuple
import random

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
    def create_with_image(cls, image) -> 'Histogram':
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
            count = random.Random(seed + 2).randint(min_count, max_count)
            hist[colors[color_index]] = count
        return cls(hist)

    def sorted_color_count_list(self) -> List[Tuple[int, int]]:
        """
        sort by popularity, if there is a tie, sort by color
        """

        items = sorted(self.color_count.items(), key=lambda item: (-item[1], item[0]))
        return items

    def pretty(self) -> str:
        histogram = self.clone()
        histogram.purge_mutable()
        color_count_list = histogram.sorted_color_count_list()
        if len(color_count_list) == 0:
            return 'empty'
        return ','.join([f'{color}:{count}' for color, count in color_count_list])

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
        find the set of colors that are in both histograms
        """
        return set(self.color_count.keys()) & set(other.color_count.keys())

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
    
    def number_of_unique_colors(self):
        """
        Number of unique colors in the histogram.
        """
        histogram = self.clone()
        histogram.purge_mutable()
        return len(histogram.color_count)

    def purge_mutable(self):
        """
        Remove colors where the count is less than 1.
        It doesn't make sense having zero counters in the histogram.
        It doesn't make sense having negative counters in the histogram.
        """
        for color in list(self.color_count.keys()):
            if self.color_count[color] < 1:
                del self.color_count[color]
    
    