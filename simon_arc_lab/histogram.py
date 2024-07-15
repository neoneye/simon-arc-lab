from typing import Dict, List, Tuple

class Histogram:
    def __init__(self, color_count: Dict[int, int]):
        self.color_count = color_count

    def clone(self) -> 'Histogram':
        """
        Create a copy of the current Histogram instance.
        """
        return Histogram(self.color_count.copy())
    
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
        color_count_list = self.sorted_color_count_list()
        return ','.join([f'{color}:{count}' for color, count in color_count_list])

    def add(self, other: 'Histogram') -> 'Histogram':
        result = self.clone()
        for color, count in other.color_count.items():
            if color in result.color_count:
                result.color_count[color] += count
            else:
                result.color_count[color] = count
        return result
