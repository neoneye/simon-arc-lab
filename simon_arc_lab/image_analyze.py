import numpy as np
from enum import Enum
from dataclasses import dataclass
from collections import Counter
from .histogram import Histogram

class FeatureType(Enum):
    TEST_A = "test_a"
    NUMBER_OF_UNIQUE_COLORS = "number_of_unique_colors"
    UNIQUE_COLORS = "unique_colors"
    COMPRESSED_REPRESENTATION = "compressed_representation"
    LENGTH_OF_COMPRESSED_REPRESENTATION = "length_of_compressed_representation"

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

class AnalyzeBase:
    def __init__(self):
        self.feature_to_valueset = {}

    def set_feature_to_valueset(self, feature: Feature, valueset: set):
        self.feature_to_valueset[feature] = valueset

    def resolve_feature(self, feature: Feature, fallback: any) -> set:
        if fallback == None:
            raise ValueError(f"Unknown feature type: {feature}")
        else:
            return fallback

    def get(self, feature: Feature, fallback: any) -> set:
        if feature in self.feature_to_valueset:
            return self.feature_to_valueset[feature]
        valueset = self.resolve_feature(feature, fallback)
        self.set_feature_to_valueset(feature, valueset)
        return valueset

    @classmethod
    def intersection_union_with_fallback(cls, analyze_line_list: list['AnalyzeBase'], feature: Feature, fallback: any) -> tuple:
        intersection = set()
        union = set()
        for analyze_line_index, analyze_line in enumerate(analyze_line_list):
            valueset = analyze_line.get(feature, fallback)
            if analyze_line_index == 0:
                intersection = valueset
                union = valueset
                continue
            intersection = intersection & valueset
            union = union | valueset
        return intersection, union

    @classmethod
    def intersection_union_raise(cls, analyze_line_list: list['AnalyzeBase'], feature: Feature) -> tuple:
        """
        Raise an exception if the feature cannot be resolved.
        """
        return cls.intersection_union_with_fallback(analyze_line_list, feature, fallback=None)

    @classmethod
    def intersection_union_emptyset(cls, analyze_line_list: list['AnalyzeBase'], feature: Feature) -> tuple:
        """
        If the feature cannot be resolved, then the empty set is used.
        """
        return cls.intersection_union_with_fallback(analyze_line_list, feature, fallback=set())

    @classmethod
    def counter_with_fallback(cls, analyze_line_list: list['AnalyzeBase'], feature: Feature, fallback: any) -> Counter:
        counter = Counter()
        for analyze_line in analyze_line_list:
            valueset = analyze_line.get(feature, fallback)
            items = valueset
            if isinstance(valueset, set):
                items = sorted(list(valueset))
            counter[items.__repr__()] += 1
        return counter

    @classmethod
    def counter_raise(cls, analyze_line_list: list['AnalyzeBase'], feature: Feature) -> Counter:
        """
        Raise an exception if the feature cannot be resolved.
        """
        return cls.counter_with_fallback(analyze_line_list, feature, fallback=None)

    @classmethod
    def counter_emptyset(cls, analyze_line_list: list['AnalyzeBase'], feature: Feature) -> Counter:
        """
        If the feature cannot be resolved, then the empty set is used.
        """
        return cls.counter_with_fallback(analyze_line_list, feature, fallback=set())


class AnalyzeLine(AnalyzeBase):
    def __init__(self, pixels: list[int]):
        super().__init__()
        self.pixels = pixels
        self.lazy_histogram = None
        self.lazy_pixels_without_duplicates = None

    def histogram(self) -> Histogram:
        """
        Populate a histogram from a line of pixels.
        """
        if self.lazy_histogram is not None:
            return self.lazy_histogram
        
        histogram = Histogram.empty()
        for pixel in self.pixels:
            histogram.increment(pixel)
        
        self.lazy_histogram = histogram
        return histogram
    
    def compressed_representation(self) -> list[int]:
        """
        Get a list of pixels without duplicated pixel values.
        """
        if self.lazy_pixels_without_duplicates is not None:
            return self.lazy_pixels_without_duplicates
        
        color_list = []
        last_color = None
        for color in self.pixels:
            if color == last_color:
                continue
            color_list.append(int(color))
            last_color = color

        self.lazy_pixels_without_duplicates = color_list
        return color_list

    def resolve_feature(self, feature: Feature, fallback: any) -> set:
        if feature.feature_type == FeatureType.NUMBER_OF_UNIQUE_COLORS:
            value = self.histogram().number_of_unique_colors()
            return {value}
        if feature.feature_type == FeatureType.UNIQUE_COLORS:
            value = self.histogram().unique_colors()
            # cast from np.uint8 to int
            value = set(map(int, value))
            return set(value)
        if feature.feature_type == FeatureType.COMPRESSED_REPRESENTATION:
            value = self.compressed_representation()
            valueset = set()
            valueset.add(tuple(value))
            return valueset
        if feature.feature_type == FeatureType.LENGTH_OF_COMPRESSED_REPRESENTATION:
            value = len(self.compressed_representation())
            return {value}
        return super().resolve_feature(feature, fallback)

class ImageAnalyze:
    def __init__(self, image: np.array):
        self.image = image

    def analyze_line_for_leftright(self) -> list[AnalyzeLine]:
        analyze_line_list = []
        for y in range(self.image.shape[0]):
            pixels = self.image[y, :]
            analyze_line_list.append(AnalyzeLine(pixels))
        return analyze_line_list

    def analyze_line_for_topbottom(self) -> list[AnalyzeLine]:
        analyze_line_list = []
        for x in range(self.image.shape[1]):
            pixels = self.image[:, x]
            analyze_line_list.append(AnalyzeLine(pixels))
        return analyze_line_list

    @classmethod
    def analyze_multiple_lines(cls, analyze_line_list: list[AnalyzeLine]) -> str:
        """
        Analyze multiple lines of pixels.
        """
        features = [
            Feature(FeatureType.NUMBER_OF_UNIQUE_COLORS),
            Feature(FeatureType.UNIQUE_COLORS),
            Feature(FeatureType.LENGTH_OF_COMPRESSED_REPRESENTATION),
            Feature(FeatureType.COMPRESSED_REPRESENTATION),
        ]
        rows = []
        for feature in features:
            intersection, union = AnalyzeLine.intersection_union_emptyset(analyze_line_list, feature)
            rows.append(f"{feature}: {intersection} / {union}")
        return "\n".join(rows)
    
    def analyze(self) -> str:
        """
        Analyze the image.
        """
        str_leftright = ImageAnalyze.analyze_multiple_lines(self.analyze_line_for_leftright())
        str_topbottom = ImageAnalyze.analyze_multiple_lines(self.analyze_line_for_topbottom())
        return f"leftright:\n{str_leftright}\ntopbottom:\n{str_topbottom}"
