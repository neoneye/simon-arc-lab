from .histogram import *

# width, height
# same total pixel count
# same number of unique colors
# same count of a particular color
# distance between histograms
# sort histogram of colors, compare counters
# same orientation: portrait, landscape, square
# bigrams
# trigrams
# shape types

class ImageSimilarity:
    def __init__(self, image0: np.array, image1: np.array) -> None:
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
            self.same_histogram(),
            self.same_unique_colors(),
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
