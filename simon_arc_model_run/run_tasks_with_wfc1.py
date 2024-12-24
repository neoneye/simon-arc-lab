from datetime import datetime
import os
import sys
import random
import numpy as np
from enum import Enum
from typing import Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.show_prediction_result import show_multiple_images

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Run id: {run_id}")

path_to_arc_dataset_collection_dataset = '/Users/neoneye/git/arc-dataset-collection/dataset'
if not os.path.isdir(path_to_arc_dataset_collection_dataset):
    print(f"ARC dataset collection directory '{path_to_arc_dataset_collection_dataset}' does not exist.")
    sys.exit(1)

datasetid_groupname_pathtotaskdir_list = [
    ('ARC-AGI', 'arcagi', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data')),
    # ('ARC-AGI', 'arcagi_training', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/training')),
    # ('ARC-AGI', 'arcagi_evaluation', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/evaluation')),
    # ('arc-dataset-tama', 'tama', os.path.join(path_to_arc_dataset_collection_dataset, 'arc-dataset-tama/data')),
    # ('Mini-ARC', 'miniarc', os.path.join(path_to_arc_dataset_collection_dataset, 'Mini-ARC/data')),
    # ('ConceptARC', 'conceptarc', os.path.join(path_to_arc_dataset_collection_dataset, 'ConceptARC/data')),
    # ('ARC-AGI', 'testdata', os.path.join(PROJECT_ROOT, 'testdata', 'ARC-AGI/data')),
]

for dataset_id, groupname, path_to_task_dir in datasetid_groupname_pathtotaskdir_list:
    if not os.path.isdir(path_to_task_dir):
        print(f"path_to_task_dir directory '{path_to_task_dir}' does not exist.")
        sys.exit(1)

task_ids_of_interest = [
    '137f0df0',
]

class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

# tile types
TILE_0 = 0
TILE_1 = 1
TILE_2 = 2
TILE_3 = 3
TILE_4 = 4
TILE_5 = 5
TILE_6 = 6
TILE_7 = 7
TILE_8 = 8
TILE_9 = 9
TILE_10 = 10

# tile edge
EDGE_A = 0
EDGE_B = 1
EDGE_C = 2
EDGE_D = 3
# EDGE_25 = 4
# EDGE_02 = 5
# EDGE_015 = 6

tileRules = {
    # TILE_0: [EDGE_015, EDGE_015, EDGE_015, EDGE_015],
    # TILE_1: [EDGE_02, EDGE_02, EDGE_02, EDGE_02],
    TILE_1: [EDGE_A, EDGE_A, EDGE_A, EDGE_A],
    TILE_2: [EDGE_A, EDGE_A, EDGE_A, EDGE_A],
    TILE_5: [EDGE_A, EDGE_A, EDGE_A, EDGE_A],
    TILE_10: [EDGE_A, EDGE_A, EDGE_A, EDGE_A],
}

tileWeights = {
    # TILE_0: 1.0,
    TILE_1: 1.0,
    TILE_2: 1.0,
    TILE_5: 1.0,
    TILE_10: 1.0,
    # TILE_3: 1.0,
}

class Tile:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.possibilities = list(tileRules.keys())
        self.entropy = len(self.possibilities)
        self.neighbors = dict()

    def __repr__(self):
        return f"Tile(x={self.x}, y={self.y}, possibilities={self.possibilities})"

    def addNeighbor(self, direction: Direction, tile: 'Tile'):
        self.neighbors[direction] = tile

    def getNeighbor(self, direction) -> Optional['Tile']:
        return self.neighbors[direction]

    def getDirections(self) -> list[Direction]:
        return list(self.neighbors.keys())

    def getPossibilities(self) -> list[int]:
        return self.possibilities
    
    def set_possibilities(self, possibilities: list[int]):
        self.possibilities = possibilities
        self.entropy = len(self.possibilities)
        if self.entropy == 1:
            self.entropy = 0

    def collapse(self):
        weights = [tileWeights[possibility] for possibility in self.possibilities]
        self.possibilities = random.choices(self.possibilities, weights=weights, k=1)
        self.entropy = 0

    def constrain(self, neighborPossibilities: list[int], direction: Direction) -> bool:
        reduced = False

        if self.entropy > 0:
            connectors = []
            for neighbourPossibility in neighborPossibilities:
                connectors.append(tileRules[neighbourPossibility][direction.value])

            # check opposite side
            if direction == Direction.UP:    opposite = Direction.DOWN
            if direction == Direction.LEFT:  opposite = Direction.RIGHT
            if direction == Direction.RIGHT: opposite = Direction.LEFT
            if direction == Direction.DOWN:  opposite = Direction.UP

            for possibility in self.possibilities.copy():
                if tileRules[possibility][opposite.value] not in connectors:
                    self.possibilities.remove(possibility)
                    reduced = True

            self.entropy = len(self.possibilities)

        return reduced


class World:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        self.tileRows = []
        for y in range(self.height):
            tiles = []
            for x in range(self.width):
                tile = Tile(x, y)
                tiles.append(tile)
            self.tileRows.append(tiles)

        for y in range(self.height):
            for x in range(self.width):
                tile = self.tileRows[y][x]
                if y > 0:
                    tile.addNeighbor(Direction.UP, self.tileRows[y - 1][x])
                if x < self.width - 1:
                    tile.addNeighbor(Direction.RIGHT, self.tileRows[y][x + 1])
                if y < self.height - 1:
                    tile.addNeighbor(Direction.DOWN, self.tileRows[y + 1][x])
                if x > 0:
                    tile.addNeighbor(Direction.LEFT, self.tileRows[y][x - 1])

    def getEntropy(self, x: int, y: int) -> int:
        return self.tileRows[y][x].entropy

    def getType(self, x: int, y: int) -> int:
        return self.tileRows[y][x].possibilities[0]

    def getLowestEntropy(self) -> int:
        lowestEntropy = len(list(tileRules.keys()))
        for y in range(self.height):
            for x in range(self.width):
                tileEntropy = self.tileRows[y][x].entropy
                if tileEntropy > 0:
                    if tileEntropy < lowestEntropy:
                        lowestEntropy = tileEntropy
        return lowestEntropy

    def getTilesLowestEntropy(self) -> list[Tile]:
        lowestEntropy = len(list(tileRules.keys()))
        tileList = []

        for y in range(self.height):
            for x in range(self.width):
                tileEntropy = self.tileRows[y][x].entropy
                if tileEntropy > 0:
                    if tileEntropy < lowestEntropy:
                        tileList.clear()
                        lowestEntropy = tileEntropy
                    if tileEntropy == lowestEntropy:
                        tileList.append(self.tileRows[y][x])
        return tileList

    def waveFunctionCollapse(self):
        tilesLowestEntropy = self.getTilesLowestEntropy()

        if tilesLowestEntropy == []:
            return 0

        tileToCollapse = random.choice(tilesLowestEntropy)
        tileToCollapse.collapse()

        stack = []
        stack.append(tileToCollapse)

        while len(stack) > 0:
            # pop the tile from the stack
            tile = stack[-1]
            stack = stack[:-1]

            tilePossibilities = tile.getPossibilities()
            directions = tile.getDirections()

            for direction in directions:
                neighbor = tile.getNeighbor(direction)
                if neighbor.entropy != 0:
                    reduced = neighbor.constrain(tilePossibilities, direction)
                    if reduced == True:
                        stack.append(neighbor)    # When possibilities were reduced need to propagate further

        return 1

number_of_items_in_list = len(datasetid_groupname_pathtotaskdir_list)
for index, (dataset_id, groupname, path_to_task_dir) in enumerate(datasetid_groupname_pathtotaskdir_list):
    save_dir = f'run_tasks_result/{run_id}/{groupname}'
    print(f"Processing {index+1} of {number_of_items_in_list}. Group name '{groupname}'. Results will be saved to '{save_dir}'")

    taskset = TaskSet.load_directory(path_to_task_dir)
    taskset.keep_tasks_with_id(set(task_ids_of_interest))
    if len(taskset.tasks) == 0:
        print(f"Skipping group: {groupname}, due to no tasks to process.")
        continue

    print(f"Found {len(taskset.tasks)} tasks for group '{groupname}'")

    for task in taskset.tasks:
        print(f"Task: {task.metadata_task_id}")
    
        image = task.example_input(0)

        height, width = image.shape
        # add a gap between the tiles, these gap pixels describes what tiles can connect to what other tiles
        gap_size = 1
        pixel_distance = gap_size + 1
        width_with_gaps = width * pixel_distance - 1
        height_with_gaps = height * pixel_distance - 1

        world = World(width_with_gaps, height_with_gaps)

        if False:
            mid_x = width // 2
            mid_y = height // 2
            world.tileRows[mid_y][mid_x].possibilities = [TILE_1]
            world.tileRows[mid_y][mid_x].entropy = 0

        if True:
            for y in range(height_with_gaps):
                for x in range(width_with_gaps):
                    possibilities = [TILE_10]
                    world.tileRows[y][x].set_possibilities(possibilities)

            for y in range(height):
                for x in range(width):
                    pixel = image[y, x]
                    dest_x = x * pixel_distance
                    dest_y = y * pixel_distance
                    if pixel == 5:
                        possibilities = [TILE_5]
                        world.tileRows[dest_y][dest_x].set_possibilities(possibilities)
                    else:
                        possibilities = [TILE_1, TILE_2]
                        world.tileRows[dest_y][dest_x].set_possibilities(possibilities)

        print(f"before collapse. world.entropy: {world.getLowestEntropy()}")
        wfc_status = world.waveFunctionCollapse()
        # print(f"Wave function collapse status: {wfc_status}")
        print(f"after collapse. world.entropy: {world.getLowestEntropy()}")

        new_image = np.zeros((height_with_gaps, width_with_gaps), dtype=np.uint8)
        for y in range(height_with_gaps):
            for x in range(width_with_gaps):
                new_image[y, x] = world.getType(x, y)
        
        title_image_list = [
            ('arc', 'input', image),
            ('arc', 'output', new_image),
        ]
        title = f'X'
        show_multiple_images(title_image_list, title=title)
