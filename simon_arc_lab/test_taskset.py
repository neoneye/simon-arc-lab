from typing import Optional
import unittest

from simon_arc_lab.task import Task
from .taskset import TaskSet

class TestTaskSet(unittest.TestCase):
    def test_load_kaggle_arcprize2024_json(self):
        taskset = TaskSet.load_kaggle_arcprize2024_json('testdata/kaggle-arc-prize-2024-challenges-truncated.json')
        actual = taskset.task_ids()
        expected = ["007bbfb7", "3428a4f5"]
        self.assertEqual(actual, expected)

    def test_load_kaggle_arcprize2025_json(self):
        taskset1 = TaskSet.load_kaggle_arcprize2024_json('testdata/kaggle-arc-prize-2025/arc-agi_training_challenges.json')
        self.assertEqual(len(taskset1.task_ids()), 1000)
        taskset2 = TaskSet.load_kaggle_arcprize2024_json('testdata/kaggle-arc-prize-2025/arc-agi_evaluation_challenges.json')
        self.assertEqual(len(taskset2.task_ids()), 120)
        taskset3 = TaskSet.load_kaggle_arcprize2024_json('testdata/kaggle-arc-prize-2025/arc-agi_test_challenges.json')
        self.assertEqual(len(taskset3.task_ids()), 240)

    def test_arcprize2025_test_without_output(self):
        # Arrange & Act
        taskset = TaskSet.load_kaggle_arcprize2024_json('testdata/kaggle-arc-prize-2025/arc-agi_test_challenges.json')
        task_without_test_output: Optional[Task] = taskset.find_task_by_id('00576224')
        self.assertIsNotNone(task_without_test_output)
        self.assertEqual(task_without_test_output.total_pixel_count(), 84)
        self.assertEqual(task_without_test_output.count(), 3)
        self.assertEqual(task_without_test_output.count_tests, 1)
        self.assertIsNotNone(task_without_test_output.test_input(0))
        # Assert
        # The output image is supposed to be None.
        self.assertIsNone(task_without_test_output.test_output(0))

    def test_repr_empty(self):
        set = TaskSet([])
        actual = repr(set)
        expected = '<TaskSet(tasks=0)>'
        self.assertEqual(actual, expected)

    def test_repr_nonempty(self):
        taskset = TaskSet.load_kaggle_arcprize2024_json('testdata/kaggle-arc-prize-2024-challenges-truncated.json')
        actual = repr(taskset)
        expected = '<TaskSet(tasks=2)>'
        self.assertEqual(actual, expected)

    def test_remove_tasks_by_id(self):
        taskset = TaskSet.load_kaggle_arcprize2024_json('testdata/kaggle-arc-prize-2024-challenges-truncated.json')
        taskset.remove_tasks_by_id({"007bbfb7"})
        actual = taskset.task_ids()
        expected = ["3428a4f5"]
        self.assertEqual(actual, expected)

    def test_keep_tasks_with_id(self):
        taskset = TaskSet.load_kaggle_arcprize2024_json('testdata/kaggle-arc-prize-2024-challenges-truncated.json')
        taskset.keep_tasks_with_id({"007bbfb7"})
        actual = taskset.task_ids()
        expected = ["007bbfb7"]
        self.assertEqual(actual, expected)
