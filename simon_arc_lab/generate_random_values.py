import random

class GenerateRandomValues:
    def __init__(self):
        self.value_constraints = []

    def append_value(self, min_value: int, max_value: int):
        self.value_constraints.append((min_value, max_value))
    
    def find_random_values(self, seed: int, max_sum: int) -> list[int]:
        for retry_index in range(100):
            random_values = []
            available = max_sum
            for contraint_index, (min_value, max_value) in enumerate(self.value_constraints):
                random_value = random.Random(seed + retry_index + contraint_index * 1000).randint(min_value, max_value)
                available -= random_value
                random_values.append(random_value)
            if available >= 0:
                return random_values
        raise Exception("Failed to find random values.")
