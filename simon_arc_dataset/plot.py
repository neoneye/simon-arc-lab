import matplotlib.pyplot as plt
from collections import Counter

def plot_prompt_length_distribution(dataset_items: list[dict]):
    if not isinstance(dataset_items, list):
        raise ValueError('dataset_items must be a list of dictionaries')

    counter = Counter()
    for item in dataset_items:
        instruction = item['instruction']
        input = item['input']
        length = len(instruction) + len(input) + 1
        counter[length] += 1

    instructions = list(counter.keys())
    counts = list(counter.values())

    plt.bar(instructions, counts)
    plt.xticks(rotation=90)
    plt.title('Prompt length distribution')
    plt.show()

def plot_response_length_distribution(dataset_items: list[dict]):
    if not isinstance(dataset_items, list):
        raise ValueError('dataset_items must be a list of dictionaries')

    counter = Counter()
    for item in dataset_items:
        output = item['output']
        length = len(output)
        counter[length] += 1

    instructions = list(counter.keys())
    counts = list(counter.values())

    plt.bar(instructions, counts)
    plt.xticks(rotation=90)
    plt.title('Response length distribution')
    plt.show()
