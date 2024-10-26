import numpy as np
from enum import Enum
from collections import defaultdict
from functools import cached_property
from .histogram import Histogram
from .task import Task

class TaskColorProfile:
    def __init__(self, task):
        self.task = task
        self.input_histograms = []
        self.output_histograms = []
        self.input_union = None
        self.input_intersection = None
        self.output_union = None
        self.output_intersection = None
        self.color_insert_union = set()
        self.color_insert_intersection = set()
        self.color_remove_union = set()
        self.color_remove_intersection = set()
        self.has_optional_color_insert = False
        self.optional_color_insert_set = set()
        self.color_mapping = {}

        self.prepare_histograms()
        self.compute_color_insert_remove()
        self.compute_optional_color_insert()
        self.compute_color_mapping()

    def prepare_histograms(self):
        """Compute histograms for input and output images."""
        # Compute histograms for input images
        for i in range(self.task.count_examples + self.task.count_tests):
            histogram = Histogram.create_with_image(self.task.input_images[i])
            self.input_histograms.append(histogram)
        self.input_union, self.input_intersection = Histogram.union_intersection(self.input_histograms)

        # Compute histograms for output images (examples only)
        for i in range(self.task.count_examples):
            histogram = Histogram.create_with_image(self.task.output_images[i])
            self.output_histograms.append(histogram)
        self.output_union, self.output_intersection = Histogram.union_intersection(self.output_histograms)

    def compute_color_insert_remove(self):
        """
        Compute color insertions and removals between inputs and outputs.
        
        Examples:
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=2bcee788
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=54d9e175
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=7b6016b9
        """
        color_insert_union = set()
        color_insert_intersection = set()
        color_remove_union = set()
        color_remove_intersection = set()

        for i in range(self.task.count_examples):
            input_colors = self.input_histograms[i].unique_colors_set()
            output_colors = self.output_histograms[i].unique_colors_set()
            color_insert = output_colors - input_colors
            color_remove = input_colors - output_colors

            color_insert_union |= color_insert
            color_remove_union |= color_remove

            if i == 0:
                color_insert_intersection = color_insert.copy()
            else:
                color_insert_intersection &= color_insert

            if i == 0:
                color_remove_intersection = color_remove.copy()
            else:
                color_remove_intersection &= color_remove

        self.color_insert_union = color_insert_union
        self.color_insert_intersection = color_insert_intersection
        self.color_remove_union = color_remove_union
        self.color_remove_intersection = color_remove_intersection

    def compute_optional_color_insert(self):
        """
        Determine if a color sometimes gets inserted.
        Where some outputs have a particular color, and others outputs does not.

        Examples:
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=253bf280
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=44d8ac46
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=44f52bb0
        """
        color_insert_difference = self.color_insert_union - self.color_insert_intersection
        optional_color_insert_set = set()
        has_optional_color_insert = False
        if len(color_insert_difference) in [1, 2]:
            optional_color_insert_set = color_insert_difference
            has_optional_color_insert = True
        self.optional_color_insert_set = optional_color_insert_set
        self.has_optional_color_insert = has_optional_color_insert

    def compute_color_mapping(self):
        """
        Determines if there a color mapping between input and output histograms

        Examples:
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=6ea4a07e
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=913fb3ed
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=54d9e175
        """
        color_mapping = {}
        consistent = True
        for i in range(self.task.count_examples):
            input_colors = self.input_histograms[i].unique_colors_set()
            output_colors = self.output_histograms[i].unique_colors_set()
            key = frozenset(input_colors)
            if key in color_mapping:
                if color_mapping[key] != output_colors:
                    consistent = False  # Inconsistent mapping. This puzzle is probably not using color mapping.
                    break
            else:
                color_mapping[key] = output_colors
        if consistent:
            self.color_mapping = color_mapping
        else:
            self.color_mapping = {}  # Reset to empty if inconsistent

    @cached_property
    def same_histogram_for_input_output(self):
        """
        Check if input and output have the same exact same histogram for each example.
        
        Examples:
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=6150a2bd
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=952a094c
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e9afcf9a
        """
        for i in range(self.task.count_examples):
            input_histogram = self.input_histograms[i]
            output_histogram = self.output_histograms[i]
            if input_histogram != output_histogram:
                return False
        return True

    @cached_property
    def same_unique_colors_for_all_outputs(self):
        """
        Check if all example outputs agree on the same unique colors.
        
        Examples:
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e179c5f4
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=dbc1a6ce
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=db93a21d
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=db3e9e38
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=d364b489
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=d2abd087
        """
        return self.output_union == self.output_intersection

    @cached_property
    def same_unique_colors_for_input_output(self):
        """Check if input and output have the same unique colors for each example."""
        for i in range(self.task.count_examples):
            input_colors = self.input_histograms[i].unique_colors_set()
            output_colors = self.output_histograms[i].unique_colors_set()
            if input_colors != output_colors:
                return False
        return True

    @property
    def has_color_insert(self):
        """Check if all the examples agree that there are are 1 or more colors inserted."""
        return len(self.color_insert_intersection) > 0

    @property
    def has_color_remove(self):
        """Check if all the examples agree that there are are 1 or more colors removed."""
        return len(self.color_remove_intersection) > 0

    @cached_property
    def most_popular_colors_of_input_are_present_in_output(self):
        """
        Check if the most popular colors of the input are present in the output for each example.

        Examples:
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=9565186b
        """
        for i in range(self.task.count_examples):
            input_histogram = self.input_histograms[i]
            output_colors = self.output_histograms[i].unique_colors_set()
            special_colors = set(input_histogram.most_popular_color_list())
            if not special_colors.issubset(output_colors):
                return False
        return True

    @cached_property
    def most_popular_colors_of_input_are_not_present_in_output(self):
        """
        Check if the most popular colors of the input are not present in the output for each example.

        Examples:
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e9b4f6fc
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=ca8de6ea
        """
        for i in range(self.task.count_examples):
            input_histogram = self.input_histograms[i]
            output_colors = self.output_histograms[i].unique_colors_set()
            special_colors = set(input_histogram.most_popular_color_list())
            if special_colors.issubset(output_colors):
                return False
        return True

    @cached_property
    def least_popular_colors_of_input_are_present_in_output(self):
        """
        Check if the least popular colors of the input are present in the output for each example.

        Examples:
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=50aad11f
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=5289ad53
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=5a5a2103
        """
        for i in range(self.task.count_examples):
            input_histogram = self.input_histograms[i]
            output_colors = self.output_histograms[i].unique_colors_set()
            special_colors = set(input_histogram.least_popular_color_list())
            if not special_colors.issubset(output_colors):
                return False
        return True

    @cached_property
    def least_popular_colors_of_input_are_not_present_in_output(self):
        """
        Check if the least popular colors of the input are not present in the output for each example.

        Examples:
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=0a2355a6
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=37d3e8b2
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=604001fa
        """
        for i in range(self.task.count_examples):
            input_histogram = self.input_histograms[i]
            output_colors = self.output_histograms[i].unique_colors_set()
            special_colors = set(input_histogram.least_popular_color_list())
            if special_colors.issubset(output_colors):
                return False
        return True

    @cached_property
    def inbetween_colors_of_input_are_present_in_output(self):
        """
        Determines if the in-between colors of the input are present in the output.

        Examples:
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=3ee1011a
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=93b4f4b3
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=94133066
        """
        for i in range(self.task.count_examples):
            inbetween_histogram = self.input_histograms[i].histogram_without_mostleast_popular_colors()
            special_colors = inbetween_histogram.unique_colors_set()
            if not special_colors:
                return False
            output_colors = self.output_histograms[i].unique_colors_set()
            if not special_colors.issubset(output_colors):
                return False
        return True

    @cached_property
    def inbetween_colors_of_input_are_not_present_in_output(self):
        """
        Determines if the in-between colors of the input are not present in the output.

        Examples:
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=aabf363d
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=ddf7fa4f
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=cf98881b
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=7c008303
        """
        for i in range(self.task.count_examples):
            inbetween_histogram = self.input_histograms[i].histogram_without_mostleast_popular_colors()
            special_colors = inbetween_histogram.unique_colors_set()
            if not special_colors:
                return False
            output_colors = self.output_histograms[i].unique_colors_set()
            if special_colors.issubset(output_colors):
                return False
        return True

    @cached_property
    def output_colors_is_subset_input_colors(self):
        """
        Check if output colors are a subset of input colors for each example.
        
        Examples:
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=e7b06bea
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=de493100
        """
        for i in range(self.task.count_examples):
            input_colors = self.input_histograms[i].unique_colors_set()
            output_colors = self.output_histograms[i].unique_colors_set()
            if not output_colors.issubset(input_colors):
                return False
        return True

    @cached_property
    def output_colors_is_subset_input_colors_with_insert_remove(self):
        """
        Determines if the output colors are a subset of the input colors with insert/remove.

        Examples:
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=9565186b
        """
        if not self.color_insert_intersection and not self.color_remove_intersection:
            return False
        for i in range(self.task.count_examples):
            input_colors = self.input_histograms[i].unique_colors_set()
            output_colors = self.output_histograms[i].unique_colors_set()
            predicted_colors = (input_colors | self.color_insert_intersection) - self.color_remove_intersection
            if not output_colors.issubset(predicted_colors):
                return False
        return True

    @cached_property
    def output_colors_is_subset_example_output_union(self):
        """
        Check if output colors are a subset of the union of example output colors.
        
        Examples:
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=0d3d703e
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=44f52bb0
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=27a28665
        - https://neoneye.github.io/arc/edit.html?dataset=ARC&task=995c5fa3
        """
        for i in range(self.task.count_examples):
            output_colors = self.output_histograms[i].unique_colors_set()
            if not output_colors.issubset(self.output_union):
                return False
        return True

    @cached_property
    def output_colors_is_subset_inputcolors_union_outputintersectioncolors(self):
        """
        Determines if the output colors are a subset of (input_colors UNION example_output_intersection).

        Examples:
        - https://neoneye.github.io/arc/edit.html?dataset=ConceptARC&task=Center4
        - https://neoneye.github.io/arc/edit.html?dataset=ConceptARC&task=ExtractObjects6
        - https://neoneye.github.io/arc/edit.html?dataset=ConceptARC&task=FilledNotFilled7
        - https://neoneye.github.io/arc/edit.html?dataset=Mini-ARC&task=find_the_most_frequent_color_for_every_2x2_l6ad7ge3gc5rtysj7p
        """
        for i in range(self.task.count_examples):
            input_colors = self.input_histograms[i].unique_colors_set()
            output_colors = self.output_histograms[i].unique_colors_set()
            predicted_colors = input_colors | self.output_intersection
            if not output_colors.issubset(predicted_colors):
                return False
        return True

    @cached_property
    def output_colors_is_subset_inputcolors_union_optionaloutputintersectioncolors(self):
        """
        Determines if the output colors are a subset of (input_colors UNION optional_output_intersection).

        Examples:
        - https://neoneye.github.io/arc/edit.html?dataset=RE-ARC-easy&task=f2829549
        """
        if not self.optional_color_insert_set:
            return False
        for i in range(self.task.count_examples):
            input_colors = self.input_histograms[i].unique_colors_set()
            output_colors = self.output_histograms[i].unique_colors_set()
            predicted_colors = input_colors | self.optional_color_insert_set
            if not output_colors.issubset(predicted_colors):
                return False
        return True

class Metric(Enum):
    SAME_HISTOGRAM_FOR_INPUT_OUTPUT = 'same_histogram_for_input_output'
    SAME_UNIQUE_COLORS_FOR_ALL_OUTPUTS = 'same_unique_colors_for_all_outputs'
    SAME_UNIQUE_COLORS_FOR_INPUT_OUTPUT = 'same_unique_colors_for_input_output'
    SAME_INSERT_REMOVE = 'same_insert_remove'
    OUTPUT_COLORS_IS_SUBSET_INPUT_COLORS = 'output_colors_is_subset_input_colors'
    OUTPUT_COLORS_IS_SUBSET_INPUT_COLORS_WITH_INSERT_REMOVE = 'output_colors_is_subset_input_colors_with_insert_remove'
    COLOR_MAPPING = 'color_mapping'
    OUTPUT_COLORS_IS_SUBSET_EXAMPLE_OUTPUT_UNION = 'output_colors_is_subset_example_output_union'
    OUTPUT_COLORS_IS_SUBSET_INPUTCOLORS_UNION_OUTPUTINTERSECTIONCOLORS = 'output_colors_is_subset_inputcolors_union_outputintersectioncolors'
    OUTPUT_COLORS_IS_SUBSET_INPUTCOLORS_UNION_OPTIONALOUTPUTINTERSECTIONCOLORS = 'output_colors_is_subset_inputcolors_union_optionaloutputintersectioncolors'
    MOST_POPULAR_COLORS_OF_INPUT_ARE_PRESENT_IN_OUTPUT = 'most_popular_colors_of_input_are_present_in_output'
    MOST_POPULAR_COLORS_OF_INPUT_ARE_NOT_PRESENT_IN_OUTPUT = 'most_popular_colors_of_input_are_not_present_in_output'
    LEAST_POPULAR_COLORS_OF_INPUT_ARE_PRESENT_IN_OUTPUT = 'least_popular_colors_of_input_are_present_in_output'
    LEAST_POPULAR_COLORS_OF_INPUT_ARE_NOT_PRESENT_IN_OUTPUT = 'least_popular_colors_of_input_are_not_present_in_output'
    INBETWEEN_COLORS_OF_INPUT_ARE_PRESENT_IN_OUTPUT = 'inbetween_colors_of_input_are_present_in_output'
    INBETWEEN_COLORS_OF_INPUT_ARE_NOT_PRESENT_IN_OUTPUT = 'inbetween_colors_of_input_are_not_present_in_output'

    def format_with_value(self, value: int) -> str:
        suffix = ''
        if self == Metric.OUTPUT_COLORS_IS_SUBSET_INPUT_COLORS:
            suffix = ' weak'
        elif self == Metric.OUTPUT_COLORS_IS_SUBSET_EXAMPLE_OUTPUT_UNION:
            suffix = ' weak'
        return f"{self.name.lower()}: {value}{suffix}"

class BenchmarkTaskColorProfile:
    def __init__(self, verbose: bool = True):
        self.count_full_correct = defaultdict(int)
        self.count_full_incorrect = defaultdict(int)
        self.count_label_correct = defaultdict(int)
        self.count_label_incorrect = defaultdict(int)
        self.count_issue = 0
        self.verbose = verbose
    
    def track_full(self, metric: Metric, value: bool):
        if value:
            self.count_full_correct[metric] += 1
        else:
            self.count_full_incorrect[metric] += 1
    
    def check_and_track_full(self, metric: Metric, condition: bool) -> bool:
        self.track_full(metric, condition)
        return condition

    def track_label(self, metric: Metric, value: bool):
        if value:
            self.count_label_correct[metric] += 1
        else:
            self.count_label_incorrect[metric] += 1
    
    def measure_task(self, task: Task):
        profile = TaskColorProfile(task)
        for test_index in range(task.count_tests):
            self.measure_test(profile, task, test_index)

    def measure_test(self, profile: TaskColorProfile, task: Task, test_index: int):
        input_image = task.test_input(test_index)
        output_image = task.test_output(test_index)
        input_histogram = Histogram.create_with_image(input_image)
        output_histogram = Histogram.create_with_image(output_image)
        self.measure_histograms(profile, task, test_index, input_histogram, output_histogram)

    def measure_histograms(self, profile: TaskColorProfile, task: Task, test_index: int, input_histogram: Histogram, output_histogram: Histogram):
        if profile.most_popular_colors_of_input_are_present_in_output:
            special_colors = set(input_histogram.most_popular_color_list())
            output_colors = output_histogram.unique_colors_set()
            correct = special_colors.issubset(output_colors)
            self.track_label(Metric.MOST_POPULAR_COLORS_OF_INPUT_ARE_PRESENT_IN_OUTPUT, correct)

        if profile.most_popular_colors_of_input_are_not_present_in_output:
            special_colors = set(input_histogram.most_popular_color_list())
            output_colors = output_histogram.unique_colors_set()
            correct = special_colors.issubset(output_colors) == False
            self.track_label(Metric.MOST_POPULAR_COLORS_OF_INPUT_ARE_NOT_PRESENT_IN_OUTPUT, correct)

        if profile.least_popular_colors_of_input_are_present_in_output:
            special_colors = set(input_histogram.least_popular_color_list())
            output_colors = output_histogram.unique_colors_set()
            correct = special_colors.issubset(output_colors)
            self.track_label(Metric.LEAST_POPULAR_COLORS_OF_INPUT_ARE_PRESENT_IN_OUTPUT, correct)

        if profile.least_popular_colors_of_input_are_not_present_in_output:
            special_colors = set(input_histogram.least_popular_color_list())
            output_colors = output_histogram.unique_colors_set()
            correct = special_colors.issubset(output_colors) == False
            self.track_label(Metric.LEAST_POPULAR_COLORS_OF_INPUT_ARE_NOT_PRESENT_IN_OUTPUT, correct)

        if profile.inbetween_colors_of_input_are_present_in_output:
            special_colors = input_histogram.histogram_without_mostleast_popular_colors().unique_colors_set()
            output_colors = output_histogram.unique_colors_set()
            correct = special_colors.issubset(output_colors)
            self.track_label(Metric.INBETWEEN_COLORS_OF_INPUT_ARE_PRESENT_IN_OUTPUT, correct)

        if profile.inbetween_colors_of_input_are_not_present_in_output:
            special_colors = input_histogram.histogram_without_mostleast_popular_colors().unique_colors_set()
            output_colors = output_histogram.unique_colors_set()
            correct = special_colors.issubset(output_colors) == False
            self.track_label(Metric.INBETWEEN_COLORS_OF_INPUT_ARE_NOT_PRESENT_IN_OUTPUT, correct)

        if profile.same_histogram_for_input_output:
            correct = output_histogram == input_histogram
            if self.check_and_track_full(Metric.SAME_HISTOGRAM_FOR_INPUT_OUTPUT, correct):
                return

        if profile.same_unique_colors_for_all_outputs:
            correct = output_histogram.unique_colors_set() == profile.output_intersection
            if self.check_and_track_full(Metric.SAME_UNIQUE_COLORS_FOR_ALL_OUTPUTS, correct):
                return
        
        if profile.same_unique_colors_for_input_output:
            correct = output_histogram.unique_colors_set() == input_histogram.unique_colors_set()
            if self.check_and_track_full(Metric.SAME_UNIQUE_COLORS_FOR_INPUT_OUTPUT, correct):
                return
        
        if profile.has_color_insert or profile.has_color_remove or profile.has_optional_color_insert:
            predicted_colors = input_histogram.unique_colors_set()
            if profile.has_color_insert:
                predicted_colors = predicted_colors | profile.color_insert_intersection
            if profile.has_color_remove:
                predicted_colors = predicted_colors - profile.color_remove_intersection
            if output_histogram.unique_colors_set() == predicted_colors:
                self.count_full_correct[Metric.SAME_INSERT_REMOVE] += 1
                return
            if profile.has_optional_color_insert:
                predicted_colors2 = predicted_colors | profile.optional_color_insert_set
                if output_histogram.unique_colors_set() == predicted_colors2:
                    self.count_full_correct[Metric.SAME_INSERT_REMOVE] += 1
                    return
            self.count_full_incorrect[Metric.SAME_INSERT_REMOVE] += 1

        if profile.output_colors_is_subset_input_colors:
            correct = output_histogram.unique_colors_set().issubset(input_histogram.unique_colors_set())
            if self.check_and_track_full(Metric.OUTPUT_COLORS_IS_SUBSET_INPUT_COLORS, correct):
                return

        if profile.output_colors_is_subset_input_colors_with_insert_remove:
            input_colors = input_histogram.unique_colors_set()
            predicted_colors = (input_colors | profile.color_insert_intersection) - profile.color_remove_intersection
            correct = output_histogram.unique_colors_set().issubset(predicted_colors)
            if self.check_and_track_full(Metric.OUTPUT_COLORS_IS_SUBSET_INPUT_COLORS_WITH_INSERT_REMOVE, correct):
                return

        if profile.output_colors_is_subset_inputcolors_union_outputintersectioncolors:
            predicted_colors = input_histogram.unique_colors_set() | profile.output_intersection
            correct = output_histogram.unique_colors_set().issubset(predicted_colors)
            if self.check_and_track_full(Metric.OUTPUT_COLORS_IS_SUBSET_INPUTCOLORS_UNION_OUTPUTINTERSECTIONCOLORS, correct):
                return

        if len(profile.color_mapping) > 0:
            key = frozenset(input_histogram.unique_colors_set())
            correct = key in profile.color_mapping and output_histogram.unique_colors_set() == profile.color_mapping[key]
            if self.check_and_track_full(Metric.COLOR_MAPPING, correct):
                return
        
        if profile.output_colors_is_subset_example_output_union:
            correct = output_histogram.unique_colors_set().issubset(profile.output_union)
            if self.check_and_track_full(Metric.OUTPUT_COLORS_IS_SUBSET_EXAMPLE_OUTPUT_UNION, correct):
                return

        if profile.output_colors_is_subset_inputcolors_union_optionaloutputintersectioncolors:
            predicted_colors = input_histogram.unique_colors_set() | profile.optional_color_insert_set
            correct = output_histogram.unique_colors_set().issubset(predicted_colors)
            if self.check_and_track_full(Metric.OUTPUT_COLORS_IS_SUBSET_INPUTCOLORS_UNION_OPTIONALOUTPUTINTERSECTIONCOLORS, correct):
                return

        self.count_issue += 1
        if self.verbose:
            print(f"issue: {task.metadata_task_id} test={test_index}")

    def print_summary(self):
        print(f"Issues: {self.count_issue}, puzzles where the transformation couldn't be identified.")

        print(f"\nCorrect full:")
        sorted_counters = sorted(self.count_full_correct.items(), key=lambda x: (-x[1], x[0].name))
        for key, count in sorted_counters:
            s = key.format_with_value(count)
            print(f"  {s}")

        print(f"\nIncorrect full, where the check was triggered, but not satisfied, a false positive:")
        sorted_counters = sorted(self.count_full_incorrect.items(), key=lambda x: (-x[1], x[0].name))
        for key, count in sorted_counters:
            s = key.format_with_value(count)
            print(f"  {s}")

        print(f"\nCorrect label:")
        sorted_counters = sorted(self.count_label_correct.items(), key=lambda x: (-x[1], x[0].name))
        for key, count in sorted_counters:
            s = key.format_with_value(count)
            print(f"  {s}")

        print(f"\nIncorrect label, where the check was triggered, but not satisfied, a false positive:")
        sorted_counters = sorted(self.count_label_incorrect.items(), key=lambda x: (-x[1], x[0].name))
        for key, count in sorted_counters:
            s = key.format_with_value(count)
            print(f"  {s}")
