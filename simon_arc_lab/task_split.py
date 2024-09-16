from .task import Task

def task_split(task: Task, seed: int, max_example_pairs: int, permutation_count: int) -> list[Task]:
    """
    Split a big task into smaller tasks.

    The big task with many examples and many tests.

    The smaller task have N examples and 1 test pair.
    """
    task_copy = task.clone()

    count_tests = task.count_tests
    count_examples = task.count_examples
    
    known_json_set = set()

    # Prevent generating a task identical to the original task.
    json_original = task.to_arcagi1_json(True)
    known_json_set.add(json_original)

    smaller_tasks = []
    for i in range(count_tests):
        test_input = task.test_input(i)
        test_output = task.test_output(i)

        # Split up many images into smaller chunks.
        count_examples_clamped = min(count_examples, max_example_pairs)

        for j in range(permutation_count):
            for retry_index in range(100):
                task_copy.shuffle_examples(seed + i * 100 + j * 100000 + retry_index * 10000000)

                # Create task with the first N examples and 1 test pair.
                new_task = Task()
                new_task.metadata_task_id = task.metadata_task_id
                new_task.metadata_path = task.metadata_path
                for j in range(count_examples_clamped):
                    new_task.append_pair(task_copy.example_input(j), task_copy.example_output(j), True)
                new_task.append_pair(test_input, test_output, False)

                json = new_task.to_arcagi1_json(True)
                if json in known_json_set:
                    # Skipping duplicate task
                    continue
                known_json_set.add(json)

                smaller_tasks.append(new_task)
                break

    return smaller_tasks
