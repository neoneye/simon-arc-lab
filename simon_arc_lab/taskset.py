import json
import random
from .task import Task
from .load_tasks_from_directory import load_tasks_from_directory

class TaskSet:
    def __init__(self, tasks: list[Task]):
        self.tasks = tasks

    @classmethod
    def load_directory(cls, path_to_task_dir: str) -> 'TaskSet':
        tasks = load_tasks_from_directory(path_to_task_dir)
        return TaskSet(tasks)
    
    @classmethod
    def load_kaggle_arcprize2024_json(cls, path_to_json_file: str) -> 'TaskSet':
        json_dict = None
        with open(path_to_json_file, 'r') as f:
            json_dict = json.load(f)

        task_ids_sorted = sorted(json_dict.keys())

        tasks = []
        for task_id in task_ids_sorted:
            task_dict = json_dict[task_id]
            task = Task.create_with_arcagi1_json(task_dict)
            task.metadata_task_id = task_id
            tasks.append(task)
        return TaskSet(tasks)

    def remove_tasks_by_id(self, task_ids_to_remove: set[str], verbose: bool=False):
        """
        Remove tasks from the TaskSet by task id.

        Sometime this is useful to remove tasks that have already been solved, to avoid solving them again.
        So focus is on previously unsolved tasks.
        """
        new_tasks = []
        count_remove = 0
        for task in self.tasks:
            if task.metadata_task_id in task_ids_to_remove:
                count_remove += 1
                if verbose:
                    print(f"Removing task id: {task.metadata_task_id}")
                continue
            new_tasks.append(task)
        if verbose:
            print(f"Removed {count_remove} tasks. Remaining tasks: {len(new_tasks)}")
        self.tasks = new_tasks

    def keep_tasks_with_id(self, task_ids_to_keep: set[str], verbose: bool=False):
        """
        Keep the tasks from the TaskSet with the task ids. Remove the rest.

        Sometime this is useful to focus on a groups on a few specific tasks.
        """
        new_tasks = []
        count_remove = 0
        for task in self.tasks:
            if task.metadata_task_id in task_ids_to_keep:
                new_tasks.append(task)
                continue
            count_remove += 1
            if verbose:
                print(f"Removing task id: {task.metadata_task_id}")
        if verbose:
            print(f"Removed {count_remove} tasks. Remaining tasks: {len(new_tasks)}")
        self.tasks = new_tasks

    def task_ids(self) -> list[str]:
        return [task.metadata_task_id for task in self.tasks]

    def __repr__(self):
        return (f'<TaskSet(tasks={len(self.tasks)})>')

    def show_random_tasks(self, count: int=1, seed: int=0):
        """
        Inspect a few random tasks from the task set, to verify that the TaskSet has been loaded correctly.
        """
        random_tasks = random.Random(seed).sample(self.tasks, count)
        for task in random_tasks:
            task.show(show_grid=False)
