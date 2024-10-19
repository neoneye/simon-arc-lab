import unittest
from .taskset import TaskSet

class TestTaskSet(unittest.TestCase):
    def test_load_kaggle_arcprize2024_json(self):
        taskset = TaskSet.load_kaggle_arcprize2024_json('testdata/kaggle-arc-prize-2024-challenges-truncated.json')
        actual = taskset.task_ids()
        expected = ["007bbfb7", "3428a4f5"]
        self.assertEqual(actual, expected)

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
