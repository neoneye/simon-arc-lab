from typing import Dict, List, Tuple

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

    def purge_mutable(self):
        """
        Remove colors where the count is less than 1.
        It doesn't make sense having zero counters in the histogram.
        It doesn't make sense having negative counters in the histogram.
        """
        for color in list(self.color_count.keys()):
            if self.color_count[color] < 1:
                del self.color_count[color]
    
    