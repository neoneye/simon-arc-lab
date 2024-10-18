import os
import sys
from pathlib import Path

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

def process_resultsjsonl_files(paths: list, minimum_score: int) -> list:
    """
    Traverse results.jsonl files.
    Identify the rows with a score >= minimum_score.
    """
    feature_data = []
    for path in paths:
        path_set = set()
        features = None
        with open(path, 'r') as file:
            lines = file.readlines()
            for line_index, line in enumerate(lines):
                data = json.loads(line)
                if line_index == 0:
                    features = data["features"]
                field_path = data["path"]
                score = int(data["score"])
                if score >= minimum_score:
                    path_set.add(field_path)

        feature_data.append({
            "features": features,
            "path_set": path_set,
        })
    return feature_data

def extract_all_paths_from_resultsjsonl_files(paths: list) -> set:
    """
    Extract all the "path" fields from the results.jsonl files.
    """
    accumulated_paths = set()
    for path in paths:
        with open(path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                data = json.loads(line)
                field_path = data["path"]
                accumulated_paths.add(field_path)

    return accumulated_paths

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
        
    return selected_sets, covered

def analyze_with_limit(paths, minimum_score: int, title: str):
    print(f"# {title} - minimum score: {minimum_score}")
    paths = find_resultsjsonl_files(analyze_dir)
    feature_data = process_resultsjsonl_files(paths, minimum_score)
    sets = []
    for data in feature_data:
        correct_path_set = data["path_set"]
        sets.append(correct_path_set)

    for number_of_sets in range(1, len(sets)+1):
        selected_sets, covered = greedy_maximum_coverage(sets, number_of_sets)
        print(f"Score: {minimum_score}, Number of sets: {number_of_sets}, covered: {len(covered)}")

def print_filenames_with_barely_any_coverage(paths, minimum_score: int, number_of_sets: int):
    paths = find_resultsjsonl_files(analyze_dir)
    feature_data = process_resultsjsonl_files(paths, minimum_score)
    sets = []
    for data in feature_data:
        correct_path_set = data["path_set"]
        sets.append(correct_path_set)

    all_paths = extract_all_paths_from_resultsjsonl_files(paths)
    print(f"Total number of paths: {len(all_paths)}")

    selected_sets, covered = greedy_maximum_coverage(sets, number_of_sets)
    print(f"Score: {minimum_score}, Number of sets: {number_of_sets}, covered: {len(covered)}")

    remaining_paths = all_paths - covered
    print(f"Remaining paths: {len(remaining_paths)}")
    sorted_paths = sorted(list(remaining_paths))
    for path in sorted_paths:
        filename = Path(path).stem
        print(filename)

def identify_filenames_with_maximum_coverage(paths: list, minimum_score: int, number_of_sets: int, save_csv_file: str):
    paths = find_resultsjsonl_files(analyze_dir)
    feature_data = process_resultsjsonl_files(paths, minimum_score)
    sets = []
    for data in feature_data:
        correct_path_set = data["path_set"]
        sets.append(correct_path_set)

    selected_sets, covered = greedy_maximum_coverage(sets, number_of_sets)
    #print(f"Score: {minimum_score}, Number of sets: {number_of_sets}, covered: {len(covered)}")

    sorted_paths = sorted(list(covered))
    filenames = []
    for path in sorted_paths:
        filename = Path(path).stem
        # print(filename)
        filenames.append(filename)
    sorted_filenames = sorted(filenames)
    #print(sorted_filenames)

    # save as a csv file
    with open(save_csv_file, 'w') as f:
        for filename in sorted_filenames:
            f.write(f"{filename}\n")
    print(f"Saved '{save_csv_file}' with {len(sorted_filenames)} rows")

analyze_dir = f'run_tasks_result/measure_decisiontree_features/202410182042_version2'
paths = find_resultsjsonl_files(analyze_dir)
analyze_with_limit(paths, 100, 'Solution must be perfect')
analyze_with_limit(paths, 95, 'Allow near perfect solutions')
#analyze_with_limit(paths, 90, 'Allow for crappy solutions')

#print_filenames_with_barely_any_coverage(paths, 100, 9)
identify_filenames_with_maximum_coverage(paths, 100, 9, 'finetune/taskids_with_score100.csv')
identify_filenames_with_maximum_coverage(paths, 95, 11, 'finetune/taskids_with_score95_or_better.csv')
