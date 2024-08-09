import unittest
from .taskset import TaskSet

class TestTaskSet(unittest.TestCase):
    def test_load_kaggle_arcprize2024_json(self):
        set = TaskSet.load_kaggle_arcprize2024_json('testdata/kaggle-arc-prize-2024-challenges-truncated.json')
        expected = ["007bbfb7", "3428a4f5"]
        actual = set.task_ids()
        self.assertEqual(actual, expected)

if __name__ == '__main__':
    unittest.main()
