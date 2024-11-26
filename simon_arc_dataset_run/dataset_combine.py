import os
import sys
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_dataset.dataset_generator import *
from dataset_cellular_automaton import DatasetCellularAutomaton
from dataset_dilation import DatasetDilation
from dataset_erosion import DatasetErosion
from dataset_histogram import DatasetHistogram
from dataset_image import DatasetImage
from dataset_image_pair import DatasetImagePair
from dataset_mass import DatasetMass
from dataset_rle import DatasetRLE
from dataset_scale import DatasetScale
from dataset_shape import DatasetShape
from dataset_task import DatasetTask
from dataset_symmetry import DatasetSymmetry
# from dataset_solve_augment import DatasetSolveAugment
from dataset_solve_bool import DatasetSolveBool
from dataset_solve_boundingbox import DatasetSolveBoundingBox
from dataset_solve_color import DatasetSolveColor
from dataset_solve_compress import DatasetSolveCompress
from dataset_solve_count import DatasetSolveCount
from dataset_solve_cross import DatasetSolveCross
from dataset_solve_edge import DatasetSolveEdge
from dataset_solve_erosion import DatasetSolveErosion
from dataset_solve_deform import DatasetSolveDeform
from dataset_solve_flip import DatasetSolveFlip
from dataset_solve_fractal import DatasetSolveFractal
from dataset_solve_gravity import DatasetSolveGravity
from dataset_solve_grid import DatasetSolveGrid
from dataset_solve_half import DatasetSolveHalf
from dataset_solve_halfplane import DatasetSolveHalfPlane
from dataset_solve_mask import DatasetSolveMask
from dataset_solve_mass import DatasetSolveMass
from dataset_solve_outline import DatasetSolveOutline
from dataset_solve_probecolor import DatasetSolveProbeColor
from dataset_solve_ray import DatasetSolveRay
from dataset_solve_rectangle import DatasetSolveRectangle
from dataset_solve_reverse import DatasetSolveReverse
from dataset_solve_rotate import DatasetSolveRotate
from dataset_solve_scale import DatasetSolveScale
from dataset_solve_span import DatasetSolveSpan
from dataset_solve_skew import DatasetSolveSkew
from dataset_solve_symmetry import DatasetSolveSymmetry
from dataset_solve_template import DatasetSolveTemplate
from dataset_solve_translate import DatasetSolveTranslate
from dataset_solve_zindex import DatasetSolveZIndex

SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'dataset_combine.jsonl')

class CombinedDatasetGenerator(DatasetGenerator):
    def __init__(self, generators: list):
        super().__init__()
        self.generators = generators

    def generate(self, seed: int, max_num_samples=1000, max_byte_size=10*1024*1024, show: bool = False):
        max_num_samples_per_generator = max_num_samples // len(self.generators)

        generator_indexes = []
        for i in range(len(self.generators)):
            for j in range(max_num_samples_per_generator):
                generator_indexes.append(i)

        row_strings = []
        dataset_items = []
        file_size = 0
        stop = False
        total = len(generator_indexes)
        progress_index = 0
        with tqdm(total=total, desc="Generating dataset", ncols=100, unit="sample", mininterval=1.0, dynamic_ncols=True, leave=True) as pbar:
            while progress_index < total:
                if stop:
                    break
                gen_index = generator_indexes[progress_index]
                gen = self.generators[gen_index]
                generator_seed = progress_index * 181333329 + seed
                items = gen.generate_dataset_item_list(generator_seed, show)
                name_of_generator = gen.__class__.__name__
                pbar.set_description(f"{name_of_generator}")
                for item in items:
                    row_string = json.dumps(item, separators=(',', ':')) + '\n'
                    bytes = len(row_string)
                    if file_size + bytes > max_byte_size:
                        stop = True
                        break
                    if len(row_strings) >= max_num_samples:
                        stop = True
                        break
                    file_size += bytes
                    row_strings.append(row_string)
                    dataset_items.append(item)
                    pbar.update(1)
                    progress_index += 1

        if len(row_strings) != len(dataset_items):
            raise Exception("len(row_strings) != len(dataset_items)")
        
        # shuffle the row_strings and dataset_items in the same order
        indexes = list(range(len(row_strings)))
        random.Random(seed).shuffle(indexes)
        row_strings = [row_strings[i] for i in indexes]
        dataset_items = [dataset_items[i] for i in indexes]

        self.row_strings = row_strings
        self.dataset_items = dataset_items

if __name__ == "__main__":
    generator_list_not_puzzles = [
        DatasetCellularAutomaton(),
        # DatasetDilation(),
        # DatasetErosion(),
        DatasetHistogram(),
        # DatasetImage(),
        # DatasetImagePair(),
        # DatasetMass(),
        # DatasetScale(),
        # DatasetShape(),
        # DatasetSymmetry(),
        # DatasetRLE(), # no longer relevant
        # DatasetTask(),
    ]
    generator_list_puzzles = [
        # DatasetSolveAugment(),
        # DatasetSolveBool(),
        # DatasetSolveBoundingBox(),
        # DatasetSolveColor(),
        # DatasetSolveCompress(),
        # DatasetSolveCount(),
        # DatasetSolveCross(),
        # DatasetSolveEdge(),
        # DatasetSolveErosion(),
        # DatasetSolveDeform(),
        # DatasetSolveFlip(),
        # DatasetSolveFractal(),
        # DatasetSolveGravity(),
        # DatasetSolveGrid(),
        # DatasetSolveHalf(),
        # DatasetSolveHalfPlane(),
        # DatasetSolveMask(),
        DatasetSolveMass(),
        DatasetSolveOutline(),
        DatasetSolveProbeColor(),
        DatasetSolveRay(),
        DatasetSolveRectangle(),
        # DatasetSolveReverse(),
        # DatasetSolveRotate(),
        # DatasetSolveScale(),
        # DatasetSolveSpan(),
        # DatasetSolveSkew(),
        # DatasetSolveSymmetry(),
        # DatasetSolveTemplate(),
        # DatasetSolveTranslate(),
        DatasetSolveZIndex(),
    ]
    generator_list = generator_list_not_puzzles + generator_list_puzzles
    generator = CombinedDatasetGenerator(generator_list)
    generator.generate(
        seed=37,
        max_num_samples=8000,
        max_byte_size=1024*1024*100,
        # show=True
    )
    generator.save(SAVE_FILE_PATH)
    generator.inspect()
