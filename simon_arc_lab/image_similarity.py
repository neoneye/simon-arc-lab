# Measure similarity between 2 images, despite having different sizes, they may still be similar.
#
# Interesting are the ARC-AGI puzzles with a tiny intersection across the pairs
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=3194b014
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=1a2e2828
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=0a1d4ef5
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=f5aa3634
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e66aafb8
#
# Interesting are the ARC-AGI puzzles with an unusual big standard deviation
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=4b6b68e5
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=54d9e175
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=a9f96cdd
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=44d8ac46
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=253bf280
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e9bb6954
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e9ac8c9e
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e1baa8a4
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=d931c21c
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=d4b1c2b1
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=1c02dbbe
#
# Interesting are the ARC-AGI puzzles with a very low similarity score
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=8731374e
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=995c5fa3
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=b9b7f026
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=2037f2c7
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=c3202e5a
#
# IDEA: does one image contain the other image, original/rotated/flipped, by checking if the all the bigrams are contained in the other image bigrams: 
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=f5aa3634
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=f4081712
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e66aafb8
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e633a9e5
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=de493100
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=d56f2372
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=cd3c21df
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=be03b35f
#
# IDEA: are there shapes that are present in both images, as in:
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=ed74f2f2
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=c64f1187
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=bf699163
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=bbb1b8b6
#
# IDEA: the mask for a particular color, is that mask present in both images, as in:
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=ea9794b1
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e9b4f6fc
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e7a25a18
#
# IDEA: same count of a particular color, is the black color the same count, as in: 
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=09c534e7
#
# IDEA: is the compressed images the same, as in:
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e1baa8a4
#
# IDEA: two images with different sizes, scaling up the histogram, are there then a color component with the same counter, as in:
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=c92b942c_v2
#
# IDEA: Make another similarity class that compares images, where the order matters.
# This way I can check, is one image a subset of the other image.
#
# IDEA: A verbose jaccard_index, where I can see which features are satisfied.
#
# IDEA: has same 3x3 structure, as in:
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=44d8ac46
#
# distance between histograms
# trigrams
# shape types
#
# IDEA: there are many agree_on_color, maybe assign a lower weight, so they don't dominate the jaccard index.
# IDEA: there are many agree_on_color_with_same_counter, maybe assign a lower weight, so they don't dominate the jaccard index.
# IDEA: there are many same_bounding_box_size_of_color, maybe assign a lower weight, so they don't dominate the jaccard index.

from .histogram import *
from .image_bigram import *
from .find_bounding_box import find_bounding_box_multiple_ignore_colors
import numpy as np
from enum import Enum
from dataclasses import dataclass

class FeatureType(Enum):
    SAME_IMAGE = "same_image"
    SAME_SHAPE = "same_shape"
    SAME_SHAPE_ALLOW_ROTATION = "same_shape_allow_rotation"
    SAME_WIDTH = "same_shape_width"
    SAME_HEIGHT = "same_shape_height"
    SAME_ORIENTATION = "same_shape_orientation"
    SAME_HISTOGRAM = "same_histogram"
    SAME_UNIQUE_COLORS = "same_unique_colors"
    SAME_NUMBER_UNIQUE_COLORS = "same_number_of_unique_colors"
    SAME_HISTOGRAM_IGNORING_SCALE = "same_histogram_ignoring_scale"
    SAME_HISTOGRAM_COUNTERS = "same_histogram_counters"
    SAME_MOST_POPULAR_COLOR_LIST = "same_most_popular_color_list"
    SAME_LEAST_POPULAR_COLOR_LIST = "same_least_popular_color_list"
    AGREE_ON_COLOR = "agree_on_color"
    AGREE_ON_COLOR_WITH_SAME_COUNTER = "agree_on_color_with_same_counter"
    UNIQUE_COLORS_IS_A_SUBSET = "unique_colors_is_a_subset"
    SAME_BOUNDING_BOX_SIZE_OF_COLOR = "same_bounding_box_size_of_color"
    SAME_BIGRAMS_DIRECTION_ALL = "same_bigrams_direction_all"
    SAME_BIGRAMS_DIRECTION_LEFTRIGHT = "same_bigrams_direction_leftright"
    SAME_BIGRAMS_DIRECTION_TOPBOTTOM = "same_bigrams_direction_topbottom"
    SAME_BIGRAMS_DIRECTION_TOPLEFTBOTTOMRIGHT = "same_bigrams_direction_topleftbottomright"
    SAME_BIGRAMS_DIRECTION_TOPRIGHTBOTTOMLEFT = "same_bigrams_direction_toprightbottomleft"
    SAME_BIGRAMS_SUBSET = "same_bigrams_subset"

@dataclass(frozen=True)
class Feature:
    feature_type: FeatureType
    parameter: any = None  # Optional parameter, e.g., color

    def __str__(self):
        if self.parameter is not None:
            return f"{self.feature_type.value}({self.parameter})"
        else:
            return self.feature_type.value
        
    @classmethod
    def format_feature_list(cls, features: list):
        feature_names = [f"{feature}" for feature in features]
        feature_names = sorted(feature_names)
        return ",".join(feature_names)

class ImageSimilarity:
    def __init__(self, image0: np.array, image1: np.array) -> None:
        """
        Compare two images. The order doesn't matter.
        """
        self.image0 = image0
        self.image1 = image1
        self.lazy_histogram0 = None
        self.lazy_histogram1 = None
        self.lazy_features = None
    
    @classmethod
    def compute_jaccard_index(cls, parameters: list[bool]) -> int:
        """
        Jaccard index of how many features are satisfied.
        
        return: 0 to 100
        """
        a_union_b = len(parameters)
        if a_union_b == 0:
            # Avoid division by zero
            # No features to compare, thus no features can be satisfied.
            return 0
        a_intersection_b = sum(parameters)
        return a_intersection_b * 100 // a_union_b

    def _compute_features(self) -> dict:
        """
        Compare features between the two images and return a dictionary
        mapping Feature instances to booleans indicating if they are satisfied.
        """
        features = {
            Feature(FeatureType.SAME_IMAGE): self.same_image(),
            Feature(FeatureType.SAME_SHAPE): self.same_shape(),
            Feature(FeatureType.SAME_SHAPE_ALLOW_ROTATION): self.same_shape_allow_for_rotation(),
            Feature(FeatureType.SAME_WIDTH): self.same_shape_width(),
            Feature(FeatureType.SAME_HEIGHT): self.same_shape_height(),
            Feature(FeatureType.SAME_ORIENTATION): self.same_shape_orientation(),
            Feature(FeatureType.SAME_HISTOGRAM): self.same_histogram(),
            Feature(FeatureType.SAME_UNIQUE_COLORS): self.same_unique_colors(),
            Feature(FeatureType.SAME_NUMBER_UNIQUE_COLORS): self.same_number_of_unique_colors(),
            Feature(FeatureType.SAME_HISTOGRAM_IGNORING_SCALE): self.same_histogram_ignoring_scale(),
            Feature(FeatureType.SAME_HISTOGRAM_COUNTERS): self.same_histogram_counters(),
            Feature(FeatureType.SAME_MOST_POPULAR_COLOR_LIST): self.same_most_popular_color_list(),
            Feature(FeatureType.SAME_LEAST_POPULAR_COLOR_LIST): self.same_least_popular_color_list(),
            Feature(FeatureType.UNIQUE_COLORS_IS_A_SUBSET): self.unique_colors_is_a_subset(),
            Feature(FeatureType.SAME_BIGRAMS_DIRECTION_ALL): self.same_bigrams_direction_all(),
            Feature(FeatureType.SAME_BIGRAMS_DIRECTION_LEFTRIGHT): self.same_bigrams_direction_leftright(),
            Feature(FeatureType.SAME_BIGRAMS_DIRECTION_TOPBOTTOM): self.same_bigrams_direction_topbottom(),
            Feature(FeatureType.SAME_BIGRAMS_DIRECTION_TOPLEFTBOTTOMRIGHT): self.same_bigrams_direction_topleftbottomright(),
            Feature(FeatureType.SAME_BIGRAMS_DIRECTION_TOPRIGHTBOTTOMLEFT): self.same_bigrams_direction_toprightbottomleft(),
            Feature(FeatureType.SAME_BIGRAMS_SUBSET): self.same_bigrams_subset(),
        }

        # Color specific features
        for color in range(10):
            features[Feature(FeatureType.AGREE_ON_COLOR, color)] = self.agree_on_color(color)
            features[Feature(FeatureType.AGREE_ON_COLOR_WITH_SAME_COUNTER, color)] = self.agree_on_color_with_same_counter(color)
            features[Feature(FeatureType.SAME_BOUNDING_BOX_SIZE_OF_COLOR, color)] = self.same_bounding_box_size_of_color(color)

        return features

    def features(self) -> dict:
        if self.lazy_features is None:
            self.lazy_features = self._compute_features()
        return self.lazy_features

    def jaccard_index(self) -> int:
        """
        Jaccard index of how many features are satisfied.

        return: 0 to 100
        """
        features = self.features()
        feature_booleans = features.values()
        return self.compute_jaccard_index(feature_booleans)

    def get_satisfied_features(self) -> list:
        """
        Return a list of feature instances that are satisfied.
        """
        features = self.features()
        return [feature for feature, satisfied in features.items() if satisfied]

    def get_unsatisfied_features(self) -> list:
        """
        Return a list of feature instances that are not satisfied.
        """
        features = self.features()
        return [feature for feature, satisfied in features.items() if not satisfied]

    def same_image(self) -> bool:
        """
        Identical images.
        """
        return np.array_equal(self.image0, self.image1)

    def same_shape(self) -> bool:
        """
        Same width and height.
        """
        return self.image0.shape == self.image1.shape
    
    def same_shape_allow_for_rotation(self) -> bool:
        """
        Same width and height, allow for rotation.
        """
        height0, width0 = self.image0.shape
        height1, width1 = self.image1.shape
        return (width0 == width1 and height0 == height1) or (width0 == height1 and height0 == width1)

    def same_shape_width(self) -> bool:
        """
        Same width
        """
        width0 = self.image0.shape[1]
        width1 = self.image1.shape[1]
        return width0 == width1

    def same_shape_height(self) -> bool:
        """
        Same height
        """
        height0 = self.image0.shape[0]
        height1 = self.image1.shape[0]
        return height0 == height1

    def same_shape_orientation(self) -> bool:
        """
        Same orientation: portrait, landscape, square.
        """
        def orientation(image: np.array) -> str:
            height, width = image.shape
            if width == height:
                return "square"
            elif width > height:
                return "landscape"
            else:
                return "portrait"
        orientation0 = orientation(self.image0)
        orientation1 = orientation(self.image1)
        return orientation0 == orientation1

    def histogram0(self) -> Histogram:
        if self.lazy_histogram0 is None:
            self.lazy_histogram0 = Histogram.create_with_image(self.image0)
        return self.lazy_histogram0
    
    def histogram1(self) -> Histogram:
        if self.lazy_histogram1 is None:
            self.lazy_histogram1 = Histogram.create_with_image(self.image1)
        return self.lazy_histogram1

    def same_histogram(self) -> bool:
        """
        Identical histogram.
        """
        histogram0 = self.histogram0()
        histogram1 = self.histogram1()
        histogram0_str = histogram0.pretty()
        histogram1_str = histogram1.pretty()
        return histogram0_str == histogram1_str

    def same_unique_colors(self) -> bool:
        """
        The same colors occur in both images.
        """
        histogram0 = self.histogram0()
        histogram1 = self.histogram1()
        colors0 = histogram0.unique_colors()
        colors1 = histogram1.unique_colors()
        return colors0 == colors1

    def same_number_of_unique_colors(self) -> bool:
        """
        Do both images have the same number of unique colors.

        One image use red+gree, the other image use blue+purple. So both images have 2 unique colors.

        Examples:
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=f8c80d96
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=f76d97a5
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e179c5f4
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=d5d6de2d
        """
        histogram0 = self.histogram0()
        histogram1 = self.histogram1()
        colors0 = histogram0.unique_colors()
        colors1 = histogram1.unique_colors()
        return len(colors0) == len(colors1)

    def same_histogram_ignoring_scale(self) -> bool:
        """
        Identical histogram, but with a different ratio.
        """
        histogram0 = self.histogram0()
        histogram1 = self.histogram1()
        height0, width0 = self.image0.shape
        height1, width1 = self.image1.shape
        mass0 = height0 * width0
        mass1 = height1 * width1

        for color in range(10):
            count0 = histogram0.color_count.get(color, 0)
            count1 = histogram1.color_count.get(color, 0)
            a = count0 * mass1
            b = count1 * mass0
            if a != b:
                return False

        return True

    def same_histogram_counters(self) -> bool:
        """
        The counters are the same in the histogram, but the colors may be different.
        """
        histogram0 = self.histogram0()
        histogram1 = self.histogram1()
        counters0 = histogram0.sorted_count_list()
        counters1 = histogram1.sorted_count_list()
        return counters0 == counters1

    def same_most_popular_color_list(self) -> bool:
        """
        Both images agree on the same most popular colors.
        """
        histogram0 = self.histogram0()
        histogram1 = self.histogram1()
        color_list0 = histogram0.most_popular_color_list()
        color_list1 = histogram1.most_popular_color_list()
        return color_list0 == color_list1

    def same_least_popular_color_list(self) -> bool:
        """
        Both images agree on the same least popular colors.
        """
        histogram0 = self.histogram0()
        histogram1 = self.histogram1()
        color_list0 = histogram0.least_popular_color_list()
        color_list1 = histogram1.least_popular_color_list()
        return color_list0 == color_list1

    def unique_colors_is_a_subset(self) -> bool:
        """
        Is the unique colors of one image a subset of the unique colors of the other image.

        Examples:
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=be94b721
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=ddf7fa4f
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=de1cd16c
        """
        histogram0 = self.histogram0()
        histogram1 = self.histogram1()
        color_set0 = set(histogram0.unique_colors())
        color_set1 = set(histogram1.unique_colors())
        a = color_set0.issubset(color_set1)
        b = color_set1.issubset(color_set0)
        return a or b

    def agree_on_color(self, color: int) -> bool:
        """
        True if both images have the specified color in their histogram.
        True if both images don't have the specified color in their histogram.
        False if only one of the images have the color, and the other image doesn't have the color.

        Do the images roughly agree on the same colors.
        For each of the 10 color, is color[i] present/not present in both images
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e99362f0
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e6de6e8f_v2
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=d4c90558
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=c8b7cc0f
        """
        histogram0 = self.histogram0()
        histogram1 = self.histogram1()
        a = histogram0.get_count_for_color(color) > 0
        b = histogram1.get_count_for_color(color) > 0
        return a == b

    def agree_on_color_with_same_counter(self, color: int) -> bool:
        """
        True if both images have the specified color in their histogram, and the count is the same.
        True if both images don't have the specified color in their histogram.
        False if the counters are different.

        Example of tasks where this is satisfied:
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=ff72ca3e
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=fea12743
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=fe9372f3
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=f9a67cb5
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=f45f5ca7
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=f3e62deb
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=f3cdc58f
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=ecaa0ec1
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e88171ec
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e74e1818
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e0fb7511
        """
        histogram0 = self.histogram0()
        histogram1 = self.histogram1()
        count0 = histogram0.get_count_for_color(color)
        count1 = histogram1.get_count_for_color(color)
        return count0 == count1

    def same_bigrams_direction_all(self) -> bool:
        """
        Both images agree on the same bigrams ignoring the bigram direction.

        Example of tasks where this is satisfied:
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=ed98d772
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=ecaa0ec1
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=eb281b96
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e9afcf9a
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e40b9e2f
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e21a174a
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=dc2aa30b
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=d8c310e9
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=ccd554ac
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=cad67732
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=bd14c3bf_v2
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=ba9d41b8
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=ba97ae07
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=b1fc8b8e
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=a57f2f04
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=a406ac07
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=9ddd00f0
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=9d9215db
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=98cf29f8
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=917bccba
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=91413438
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=8ee62060
        """
        bigram0 = image_bigrams_direction_all(self.image0, 255)
        bigram1 = image_bigrams_direction_all(self.image1, 255)
        return bigram0 == bigram1

    def same_bigrams_direction_leftright(self) -> bool:
        """
        Both images agree on the same bigrams, only considering the horizontal bigrams.

        Example of tasks where this is satisfied:
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e74e1818
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e633a9e5
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e21a174a
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e1baa8a4
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=cf133acc
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=ce8d95cc
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=b9630600
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=93c31fbe
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=8719f442
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=85b81ff1
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=7ee1c6ea
        """
        bigram0 = image_bigrams_direction_leftright(self.image0, 255)
        bigram1 = image_bigrams_direction_leftright(self.image1, 255)
        return bigram0 == bigram1

    def same_bigrams_direction_topbottom(self) -> bool:
        """
        Both images agree on the same bigrams, only considering the vertical bigrams.

        Example of tasks where this is satisfied:
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=5af49b42
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=20981f0e
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e9afcf9a
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=d8c310e9
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=82819916
        """
        bigram0 = image_bigrams_direction_topbottom(self.image0, 255)
        bigram1 = image_bigrams_direction_topbottom(self.image1, 255)
        return bigram0 == bigram1

    def same_bigrams_direction_topleftbottomright(self) -> bool:
        """
        Both images agree on the same bigrams, only considering the diagonal bigrams.

        Example of tasks where this is satisfied:
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=bbc9ae5d
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=9dfd6313
        """
        bigram0 = image_bigrams_direction_topleftbottomright(self.image0, 255)
        bigram1 = image_bigrams_direction_topleftbottomright(self.image1, 255)
        return bigram0 == bigram1

    def same_bigrams_direction_toprightbottomleft(self) -> bool:
        """
        Both images agree on the same bigrams, only considering the diagonal bigrams.

        Example of tasks where this is satisfied:
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=9dfd6313
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=b8cdaf2b
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=ba97ae07
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=d10ecb37
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=feca6190
        """
        bigram0 = image_bigrams_direction_toprightbottomleft(self.image0, 255)
        bigram1 = image_bigrams_direction_toprightbottomleft(self.image1, 255)
        return bigram0 == bigram1

    def same_bigrams_subset(self) -> bool:
        """
        One image bigrams is a subset of the other image.

        Example of tasks where this is satisfied:
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=3194b014
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=642d658d
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=9a4bb226
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=bf699163
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=4938f0c2
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=44d8ac46
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=253bf280
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=22eb0ac0
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=d4b1c2b1
        """
        bigram_list0 = image_bigrams_direction_all(self.image0, 255)
        bigram_list1 = image_bigrams_direction_all(self.image1, 255)

        # remove bigrams where the tuple contains 255
        bigram_set0 = set()
        for bigram in bigram_list0:
            if 255 not in bigram:
                bigram_set0.add(bigram)

        bigram_set1 = set()
        for bigram in bigram_list1:
            if 255 not in bigram:
                bigram_set1.add(bigram)

        if len(bigram_set0) == 0 or len(bigram_set1) == 0:
            return False

        subset_a = bigram_set0.issubset(bigram_set1)
        subset_b = bigram_set1.issubset(bigram_set0)
        return subset_a or subset_b

    def same_bounding_box_size_of_color(self, color: int) -> bool:
        """
        Determines the bounding box of the specified color, 
        and checks if the bounding box size is the same in both images.

        Returns True if the bounding box is the same size in both images.
        Returns False if the bounding box is not the same size in both images.

        Example of tasks where this is satisfied:
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=f45f5ca7
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=f3e62deb
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=ff72ca3e
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=f9a67cb5
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=f83cb3f6
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e95e3d8e
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e88171ec
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e7639916
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e74e1818
        https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e41c6fd3
        """
        histogram0 = self.histogram0()
        histogram1 = self.histogram1()
        count0 = histogram0.get_count_for_color(color)
        count1 = histogram1.get_count_for_color(color)
        if count0 == 0 and count1 == 0:
            # The color is not present in any of the images.
            # Thus the bounding box is the same size, 0x0 == 0x0
            return True
        if count0 == 0 or count1 == 0:
            # The color is not present in one image, but not the other image.
            # Thus the bounding box is not the same size, 0x0 != WIDTHxHEIGHT
            return False

        ignore_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        ignore_colors.remove(color)

        rect0 = find_bounding_box_multiple_ignore_colors(self.image0, ignore_colors)
        rect1 = find_bounding_box_multiple_ignore_colors(self.image1, ignore_colors)

        same_width = rect0.width == rect1.width
        same_height = rect0.height == rect1.height
        same_size = same_width and same_height
        return same_size

