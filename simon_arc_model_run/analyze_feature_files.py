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
        correct_path_set = set()
        features = None
        with open(path, 'r') as file:
            lines = file.readlines()
            for line_index, line in enumerate(lines):
                data = json.loads(line)
                if line_index == 0:
                    features = data["features"]
                field_path = data["path"]
                if data["correct"] == True:
                    correct_count += 1
                    correct_path_set.add(field_path)

        feature_data.append({
            # "correct": correct_count,
            "features": features,
            "correct_path_set": correct_path_set,
        })
    return feature_data

def greedy_maximum_coverage(sets, k):
    """
    https://en.wikipedia.org/wiki/Maximum_coverage_problem
    """
    covered = set()
    selected_sets = []
    
    for _ in range(k):
        best_set = None
        best_coverage = 0
        
        for s in sets:
            coverage = len(s - covered)  # Elements not yet covered
            if coverage > best_coverage:
                best_coverage = coverage
                best_set = s
                
        if best_set is not None:
            selected_sets.append(best_set)
            covered.update(best_set)
        
    return selected_sets, len(covered)

analyze_dir = f'run_tasks_result/measure_decisiontree_features/202410181028'
paths = find_resultsjsonl_files(analyze_dir)
feature_data = process_resultsjsonl_files(paths)
#print(f"Feature {feature_data}")


sets = []
for data in feature_data:
    correct_path_set = data["correct_path_set"]
    sets.append(correct_path_set)

for number_of_sets in range(1, len(sets)+1):
    selected_sets, covered = greedy_maximum_coverage(sets, number_of_sets)
    print(f"Number of sets: {number_of_sets}, covered: {covered}")
#selected_sets, covered = greedy_maximum_coverage(sets, 2)
#print(f"Covered: {covered}")
#print(f"Number of items: {len(covered)}")
#print(f"Selected sets: {selected_sets}")

