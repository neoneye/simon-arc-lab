# IDEA: is the output histogram a subset of the input histogram, for all pairs.
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=be94b721
#
# IDEA: compare scale factor across the input/output images.
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=c59eb873
# https://neoneye.github.io/arc/edit.html?dataset=ARC&task=c3e719e8
#
# IDEA: score the features are there in the maybe set, with a lower weight. Or with an lower confidence level.
# IDEA: score the features are there in the outside-intersection set, with a negative weight.
#
# IDEA: compare all input histograms with each other, and determine what features are common.
# IDEA: compare example output histograms with each other, and determine what features are common.
import numpy as np
from .task import Task
from .image_with_cache import ImageWithCache
from .image_similarity import ImageSimilarity, Feature

class TaskSimilarityMultiImage:
    def __init__(self, feature_set_intersection: set, feature_set_union: set):
        self.feature_set_intersection = feature_set_intersection
        self.feature_set_union = feature_set_union

    @classmethod
    def analyze_images(cls, imagewithcache_list: list[ImageWithCache]) -> 'TaskSimilarityMultiImage':
        """
        Identify what does these images have in common.
        """

        # Schedule how the images are to be compared with each other.
        count = len(imagewithcache_list)
        comparisons = []
        for i in range(count):
            index0 = i
            for j in range(i+1, count):
                index1 = j
                comparisons.append((index0, index1))

        # Do several two-image comparisons
        feature_set_intersection = set()
        feature_set_union = set()
        for i, (index0, index1) in enumerate(comparisons):
            image_similarity = ImageSimilarity(imagewithcache_list[index0], imagewithcache_list[index1])
            feature_list = image_similarity.get_satisfied_features()

            feature_set = set(feature_list)
            if i == 0:
                feature_set_intersection = feature_set
            else:
                feature_set_intersection = feature_set & feature_set_intersection

            feature_set_union = feature_set_union | feature_set

        return cls(feature_set_intersection, feature_set_union)

class TaskSimilarity:
    def __init__(self, task: Task):
        self.task = task
        self.input_image_with_cache_list = None
        self.output_image_with_cache_list = None
        self.example_pair_feature_set_intersection = None
        self.example_pair_feature_set_union = None
        self.example_output_feature_set_intersection = None
        self.example_output_feature_set_union = None

    @classmethod
    def create_with_task(cls, task: Task) -> 'TaskSimilarity':
        ts = cls(task)
        ts.prepare_image_cache()
        ts.populate_example_output_feature_set()
        ts.populate_example_pair_feature_set()
        return ts
    
    def prepare_image_cache(self):
        """
        Allocate ImageWithCache instances for all images, except the test output images.
        """
        # The example input images and the test input images.
        input_image_with_cache_list = []
        for i in range(self.task.count_examples + self.task.count_tests):
            image = self.task.input_images[i]
            image_with_cache = ImageWithCache(image)
            input_image_with_cache_list.append(image_with_cache)

        # The example output images, excluding the test output images.
        output_image_with_cache_list = []
        for i in range(self.task.count_examples):
            image = self.task.output_images[i]
            image_with_cache = ImageWithCache(image)
            output_image_with_cache_list.append(image_with_cache)
        
        self.input_image_with_cache_list = input_image_with_cache_list
        self.output_image_with_cache_list = output_image_with_cache_list


    def populate_example_pair_feature_set(self):
        """
        Compare input/output pairs of the example pairs. Don't process the test pairs.
        """
        assert len(self.input_image_with_cache_list) == (self.task.count_examples + self.task.count_tests)
        assert len(self.output_image_with_cache_list) == self.task.count_examples

        task = self.task
        feature_set_intersection = set()
        feature_set_union = set()
        for i in range(task.count_examples):
            image_with_cache0 = self.input_image_with_cache_list[i]
            image_with_cache1 = self.output_image_with_cache_list[i]
            image_similarity = ImageSimilarity(image_with_cache0, image_with_cache1)
            feature_list = image_similarity.get_satisfied_features()

            # IDEA: pair specific features, such as some colors for that pair, and some other colors for another pair.
            # How do I transfer this knowledge to the test pair?

            # IDEA: compute intersection/union of unsatisfied features.
            # This may give a better measure of similarity of how the predicted output is different from the expected output.

            feature_set = set(feature_list)
            if i == 0:
                feature_set_intersection = feature_set
            else:
                feature_set_intersection = feature_set & feature_set_intersection

            feature_set_union = feature_set_union | feature_set

        self.example_pair_feature_set_intersection = feature_set_intersection
        self.example_pair_feature_set_union = feature_set_union

    def populate_example_output_feature_set(self):
        """
        Compare all output images of the example pairs. Don't process the test pairs.
        """
        assert len(self.output_image_with_cache_list) == self.task.count_examples
        result = TaskSimilarityMultiImage.analyze_images(self.output_image_with_cache_list)
        self.example_output_feature_set_intersection = result.feature_set_intersection
        self.example_output_feature_set_union = result.feature_set_union

    def summary(self) -> str:
        items = [
            f"output: {len(self.example_output_feature_set_intersection)}",
            f"pair: {len(self.example_pair_feature_set_intersection)}",
        ]
        return " ".join(items)

    def measure_test_prediction(self, predicted_output: np.array, test_index: int) -> int:
        """
        On Kaggle ARC-Prize, the test pairs are not provided. It's up to the solver to predict the output.
        I cannot peek at the test pair to determine if it's correct.
        This function, without knowing the `test_output`, it checks that the predicted_output
        has the same features satisfied as there are typically satisfied in the example pairs.

        returns: int, a value between 0 and 100, where 100 is the best match.
        """
        task = self.task

        predicted_output_image_with_cache = ImageWithCache(predicted_output)

        # Compare the test input image with the predicted output.
        test_input_image_with_cache = self.input_image_with_cache_list[task.count_examples + test_index]
        image_similarity = ImageSimilarity(test_input_image_with_cache, predicted_output_image_with_cache)
        feature_list = image_similarity.get_satisfied_features()
        feature_set = set(feature_list)

        parameter_list = []
        for key in self.example_pair_feature_set_intersection:
            # Are there any features that are not satisfied, for the test input vs. predicted output, 
            # it may be a sign that the prediction differs from the typical input/output features.
            is_satisfied = key in feature_set
            parameter_list.append(is_satisfied)

        # Compare all example output images with the predicted output.
        output_images = self.output_image_with_cache_list.copy()
        output_images.append(predicted_output_image_with_cache)
        result = TaskSimilarityMultiImage.analyze_images(output_images)
        for key in self.example_output_feature_set_intersection:
            # Are there any features that are not satisfied, for the predicted output, 
            # it may be a sign that the prediction differs from the example outputs.
            is_satisfied = key in result.feature_set_intersection
            parameter_list.append(is_satisfied)

        score = ImageSimilarity.compute_jaccard_index(parameter_list)
        return score
