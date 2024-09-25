from .histogram import *

# IDEA: Make another similarity class that compares images, where the order matters.
# This way I can check, is one image a subset of the other image.
#
# IDEA: A verbose jaccard_index, where I can see which features are satisfied.
#
# same total pixel count
# same number of unique colors
# same count of a particular color
# distance between histograms
# sort histogram of colors, compare counters
# bigrams
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
