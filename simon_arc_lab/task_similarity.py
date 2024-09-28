# IDEA: compare all input histograms with each other, and determine what features are common.
# IDEA: compare example output histograms with each other, and determine what features are common.
import numpy as np
from .task import Task
from .image_similarity import ImageSimilarity, Feature

class TaskSimilarity:
    def __init__(self, task: Task):
        self.task = task
        self.example_pair_feature_set_intersection = None
        self.example_pair_feature_set_union = None
        self.all_input_feature_set_intersection = None
        self.all_input_feature_set_union = None
        self.example_output_feature_set_intersection = None
        self.example_output_feature_set_union = None


    @classmethod
    def create_with_task(cls, task: Task) -> 'TaskSimilarity':
        ts = cls(task)
        ts.populate_all_input_feature_set()
        ts.populate_example_output_feature_set()
        ts.populate_example_pair_feature_set()
        return ts

    def populate_example_pair_feature_set(self):
        """
        Compare input/output pairs of the example pairs. Don't process the test pairs.
        """
        task = self.task
        feature_set_intersection = set()
        feature_set_union = set()
        for i in range(task.count_examples):
            input = task.input_images[i]
            output = task.output_images[i]
            image_similarity = ImageSimilarity(input, output)
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

    def populate_all_input_feature_set(self):
        """
        Compare all input images of the example+test pairs.
        """
        task = self.task
        count = task.count_examples + task.count_tests
        comparisons = []
        for i in range(count):
            index0 = i
            for j in range(i+1, count):
                index1 = j
                comparisons.append((index0, index1))

        feature_set_intersection = set()
        feature_set_union = set()
        for i, (index0, index1) in enumerate(comparisons):
            input0 = task.input_images[index0]
            input1 = task.input_images[index1]
            image_similarity = ImageSimilarity(input0, input1)
            feature_list = image_similarity.get_satisfied_features()

            feature_set = set(feature_list)
            if i == 0:
                feature_set_intersection = feature_set
            else:
                feature_set_intersection = feature_set & feature_set_intersection

            feature_set_union = feature_set_union | feature_set

        self.all_input_feature_set_intersection = feature_set_intersection
        self.all_input_feature_set_union = feature_set_union

    def populate_example_output_feature_set(self):
        """
        Compare all output images of the example pairs. Don't process the test pairs.
        """
        task = self.task
        count = task.count_examples
        comparisons = []
        for i in range(count):
            index0 = i
            for j in range(i+1, count):
                index1 = j
                comparisons.append((index0, index1))

        feature_set_intersection = set()
        feature_set_union = set()
        for i, (index0, index1) in enumerate(comparisons):
            input0 = task.output_images[index0]
            input1 = task.output_images[index1]
            image_similarity = ImageSimilarity(input0, input1)
            feature_list = image_similarity.get_satisfied_features()

            feature_set = set(feature_list)
            if i == 0:
                feature_set_intersection = feature_set
            else:
                feature_set_intersection = feature_set & feature_set_intersection

            feature_set_union = feature_set_union | feature_set

        self.example_output_feature_set_intersection = feature_set_intersection
        self.example_output_feature_set_union = feature_set_union

    def summary(self) -> str:
        items = [
            f"input: {len(self.all_input_feature_set_intersection)}",
            f"output: {len(self.example_output_feature_set_intersection)}",
            f"pair: {len(self.example_pair_feature_set_intersection)}",
        ]
        return " ".join(items)

    def measure_test_prediction(self, predicted_output: np.array, test_index: int) -> int:
        """
        On Kaggle ARC-Prize, the test pairs are not provided. It's up to the solver to predict the output.
        I cannot peek at the test pair to determine if it's correct.
        This function, without knowing the `test_output`, it checks that the predicted_output
        has the same features as the example pairs.

        returns: int, a value between 0 and 100, where 100 is the closest match.
        """
        task = self.task
        input = task.test_input(test_index)
        image_similarity = ImageSimilarity(input, predicted_output)
        feature_list = image_similarity.get_satisfied_features()
        feature_set = set(feature_list)

        parameter_list = []
        for key in self.example_pair_feature_set_intersection:
            is_satisfied = key in feature_set
            parameter_list.append(is_satisfied)

        score = ImageSimilarity.compute_jaccard_index(parameter_list)
        return score
