import numpy as np
from .histogram import Histogram

class ImageWithCache:
    def __init__(self, image: np.array):
        self.image = image
        self.lazy_histogram = None
        self.lazy_histogram_sorted_count_list = None
        self.lazy_histogram_most_popular_color_list = None
        self.lazy_histogram_least_popular_color_list = None
        self.lazy_histogram_unique_colors = None

    def histogram(self) -> Histogram:
        if self.lazy_histogram is None:
            self.lazy_histogram = Histogram.create_with_image(self.image)
        return self.lazy_histogram

    def histogram_sorted_count_list(self) -> list[int]:
        if self.lazy_histogram_sorted_count_list is None:
            self.lazy_histogram_sorted_count_list = self.histogram().sorted_count_list()
        return self.lazy_histogram_sorted_count_list

    def histogram_most_popular_color_list(self) -> list[int]:
        if self.lazy_histogram_most_popular_color_list is None:
            self.lazy_histogram_most_popular_color_list = self.histogram().most_popular_color_list()
        return self.lazy_histogram_most_popular_color_list

    def histogram_least_popular_color_list(self) -> list[int]:
        if self.lazy_histogram_least_popular_color_list is None:
            self.lazy_histogram_least_popular_color_list = self.histogram().least_popular_color_list()
        return self.lazy_histogram_least_popular_color_list

    def histogram_unique_colors(self) -> list[int]:
        if self.lazy_histogram_unique_colors is None:
            self.lazy_histogram_unique_colors = self.histogram().unique_colors()
        return self.lazy_histogram_unique_colors
