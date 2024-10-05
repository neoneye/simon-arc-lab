from typing import Tuple
import numpy as np
from .histogram import Histogram
from .image_bigram import *
from .image_shape2x2 import *

class ImageWithCache:
    def __init__(self, image: np.array):
        self.image = image
        self.lazy_histogram = None
        self.lazy_histogram_sorted_count_list = None
        self.lazy_histogram_most_popular_color_list = None
        self.lazy_histogram_least_popular_color_list = None
        self.lazy_histogram_unique_colors = None
        self.lazy_bigrams_direction_all = None
        self.lazy_bigrams_direction_leftright = None
        self.lazy_bigrams_direction_topbottom = None
        self.lazy_bigrams_direction_topleftbottomright = None
        self.lazy_bigrams_direction_toprightbottomleft = None
        self.lazy_shape2x2_id_list = None

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

    def bigrams_direction_all(self) -> list[Tuple[int, int]]:
        if self.lazy_bigrams_direction_all is None:
            self.lazy_bigrams_direction_all = image_bigrams_direction_all(self.image, 255)
        return self.lazy_bigrams_direction_all

    def bigrams_direction_leftright(self) -> list[Tuple[int, int]]:
        if self.lazy_bigrams_direction_leftright is None:
            self.lazy_bigrams_direction_leftright = image_bigrams_direction_leftright(self.image, 255)
        return self.lazy_bigrams_direction_leftright

    def bigrams_direction_topbottom(self) -> list[Tuple[int, int]]:
        if self.lazy_bigrams_direction_topbottom is None:
            self.lazy_bigrams_direction_topbottom = image_bigrams_direction_topbottom(self.image, 255)
        return self.lazy_bigrams_direction_topbottom

    def bigrams_direction_topleftbottomright(self) -> list[Tuple[int, int]]:
        if self.lazy_bigrams_direction_topleftbottomright is None:
            self.lazy_bigrams_direction_topleftbottomright = image_bigrams_direction_topleftbottomright(self.image, 255)
        return self.lazy_bigrams_direction_topleftbottomright

    def bigrams_direction_toprightbottomleft(self) -> list[Tuple[int, int]]:
        if self.lazy_bigrams_direction_toprightbottomleft is None:
            self.lazy_bigrams_direction_toprightbottomleft = image_bigrams_direction_toprightbottomleft(self.image, 255)
        return self.lazy_bigrams_direction_toprightbottomleft

    def shape2x2_id_list(self) -> list[int]:
        if self.lazy_shape2x2_id_list is None:
            self.lazy_shape2x2_id_list = ImageShape2x2.shape_id_list(self.image)
        return self.lazy_shape2x2_id_list
