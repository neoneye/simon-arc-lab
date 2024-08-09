import json
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

    def task_ids(self) -> list[str]:
        return [task.metadata_task_id for task in self.tasks]
