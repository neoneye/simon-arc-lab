import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from tqdm import tqdm
import json
from math import ceil
import numpy as np
import datetime
import time
import random
from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.image_pixel_similarity import image_pixel_similarity_overall
from simon_arc_model.decision_tree_util import DecisionTreeUtil, DecisionTreeFeature

def load_summaryjson_files(directory):
    paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("summary.json"):
                paths.append(os.path.join(root, file))
    paths_sorted = sorted(paths)

    feature_data = []
    for path in paths_sorted:
        with open(path) as f:
            data = json.load(f)
            feature_data.append({
                "correct": data["correct"],
                "features": data["features"],
            })
    return feature_data

def find_resultsjsonl_files(directory) -> list:
    paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("results.jsonl"):
                paths.append(os.path.join(root, file))
    return sorted(paths)

def process_resultsjsonl_files(paths: list) -> list:
    feature_data = []
    for path in paths:
        correct_count = 0
        features = None
        with open(path, 'r') as file:
            lines = file.readlines()
            for line_index, line in enumerate(lines):
                data = json.loads(line)
                if line_index == 0:
                    features = data["features"]
                if data["correct"] == True:
                    correct_count += 1

        feature_data.append({
            "correct": correct_count,
            "features": features
        })
    return feature_data

analyze_dir = f'run_tasks_result/measure_decisiontree_features/202410180010'
paths = find_resultsjsonl_files(analyze_dir)
feature_data = process_resultsjsonl_files(paths)
print(f"Feature {feature_data}")
