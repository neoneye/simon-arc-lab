import unittest
import numpy as np
from .task import *
from .task_split import *

def task_list_to_jsonl(task_list: list[Task]) -> str:
    compact = True
    return '\n'.join([task.to_arcagi1_json(compact) for task in task_list])

class TestTaskSplit(unittest.TestCase):
    def test_task_with_one_test_pair_permutation_count1(self):
        # Arrange
        task = Task()
        image0 = np.array([[0]], dtype=np.uint8)
        image1 = np.array([[1]], dtype=np.uint8)
        image2 = np.array([[2]], dtype=np.uint8)
        image3 = np.array([[3]], dtype=np.uint8)
        task.append_pair(image0, image0, True)
        task.append_pair(image1, image1, True)
        task.append_pair(image2, image2, True)
        task.append_pair(image3, image3, False)
        # Act
        task_list = task_split(task, 42, 3, 1)
        # Assert
        actual = task_list_to_jsonl(task_list)
        expected = '{"train":[{"input":[[1]],"output":[[1]]},{"input":[[0]],"output":[[0]]},{"input":[[2]],"output":[[2]]}],"test":[{"input":[[3]],"output":[[3]]}]}'
        self.assertEqual(actual, expected)

    def test_task_with_one_test_pair_permutation_count2(self):
        # Arrange
        task = Task()
        image0 = np.array([[0]], dtype=np.uint8)
        image1 = np.array([[1]], dtype=np.uint8)
        image2 = np.array([[2]], dtype=np.uint8)
        image3 = np.array([[3]], dtype=np.uint8)
        image4 = np.array([[4]], dtype=np.uint8)
        task.append_pair(image0, image0, True)
        task.append_pair(image1, image1, True)
        task.append_pair(image2, image2, True)
        task.append_pair(image3, image3, True)
        task.append_pair(image4, image4, False)
        # Act
        task_list = task_split(task, 42, 3, 2)
        # Assert
        actual = task_list_to_jsonl(task_list)
        item0 = '{"train":[{"input":[[2]],"output":[[2]]},{"input":[[1]],"output":[[1]]},{"input":[[3]],"output":[[3]]}],"test":[{"input":[[4]],"output":[[4]]}]}'
        item1 = '{"train":[{"input":[[1]],"output":[[1]]},{"input":[[3]],"output":[[3]]},{"input":[[0]],"output":[[0]]}],"test":[{"input":[[4]],"output":[[4]]}]}'
        expected = item0 + '\n' + item1
        self.assertEqual(actual, expected)

    def test_task_with_two_test_pairs(self):
        # Arrange
        task = Task()
        image0 = np.array([[0]], dtype=np.uint8)
        image1 = np.array([[1]], dtype=np.uint8)
        image2 = np.array([[2]], dtype=np.uint8)
        image3 = np.array([[3]], dtype=np.uint8)
        image4 = np.array([[4]], dtype=np.uint8)
        task.append_pair(image0, image0, True)
        task.append_pair(image1, image1, True)
        task.append_pair(image2, image2, True)
        task.append_pair(image3, image3, False) # test pair A
        task.append_pair(image4, image4, False) # test pair B
        # Act
        task_list = task_split(task, 42, 3, 2)
        # Assert
        actual = task_list_to_jsonl(task_list)
        # test pair A
        item0 = '{"train":[{"input":[[1]],"output":[[1]]},{"input":[[0]],"output":[[0]]},{"input":[[2]],"output":[[2]]}],"test":[{"input":[[3]],"output":[[3]]}]}'
        item1 = '{"train":[{"input":[[0]],"output":[[0]]},{"input":[[2]],"output":[[2]]},{"input":[[1]],"output":[[1]]}],"test":[{"input":[[3]],"output":[[3]]}]}'
        # test pair B
        item2 = '{"train":[{"input":[[0]],"output":[[0]]},{"input":[[2]],"output":[[2]]},{"input":[[1]],"output":[[1]]}],"test":[{"input":[[4]],"output":[[4]]}]}'
        item3 = '{"train":[{"input":[[1]],"output":[[1]]},{"input":[[2]],"output":[[2]]},{"input":[[0]],"output":[[0]]}],"test":[{"input":[[4]],"output":[[4]]}]}'
        expected = item0 + '\n' + item1 + '\n' + item2 + '\n' + item3
        self.assertEqual(actual, expected)
