from .histogram import *
from .image_bigram import *

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
# distance between histograms
# bigrams in the direction: horizontal, vertical, diagonal
# trigrams
# shape types

class ImageSimilarity:
    def __init__(self, image0: np.array, image1: np.array) -> None:
        """
        Compare two images. The order doesn't matter.
        """
        self.image0 = image0
        self.image1 = image1
        self.lazy_histogram0 = None
        self.lazy_histogram1 = None
    
    @classmethod
    def compute_jaccard_index(cls, parameters: list[bool]) -> int:
        """
        Jaccard index of of many features are satisfied.
        
        return: 0 to 100
        """
        a_intersection_b = 0
        for param in parameters:
            if param:
                a_intersection_b += 1
        a_union_b = len(parameters)
        return a_intersection_b * 100 // a_union_b

    def jaccard_index(self) -> int:
        """
        Jaccard index of of many features are satisfied.
        
        return: 0 to 100
        """
        if self.same_image():
            # The images are identical, no need to compute the rest of the features.
            return 100
        
        params = [
            False, # Since same_image() is False.
            self.same_shape(),
            self.same_shape_allow_for_rotation(),
            self.same_shape_width(),
            self.same_shape_height(),
            self.same_shape_orientation(),
            self.same_histogram(),
            self.same_unique_colors(),
            self.same_histogram_ignoring_scale(),
            self.same_histogram_counters(),
            self.same_most_popular_color_list(),
            self.same_least_popular_color_list(),

            # IDEA: there are many agree_on_color, maybe assign a lower weight to these in the score
            self.agree_on_color(0),
            self.agree_on_color(1),
            self.agree_on_color(2),
            self.agree_on_color(3),
            self.agree_on_color(4),
            self.agree_on_color(5),
            self.agree_on_color(6),
            self.agree_on_color(7),
            self.agree_on_color(8),
            self.agree_on_color(9),

            self.same_bigrams_direction_all(),
            self.same_bigrams_direction_leftright(),
            self.same_bigrams_direction_topbottom(),
        ]
        return self.compute_jaccard_index(params)

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

