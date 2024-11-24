from enum import Enum
import numpy as np
from math import sqrt
from simon_arc_lab.remap import remap

class DataPoint(Enum):
    PAIR_ID = 0
    PIXEL_VALUE = 1
    X = 2
    Y = 3
    WIDTH = 4
    HEIGHT = 5
    PIXEL_VALUES = 6

def datapoints_from_image(pair_id: int, image: np.array) -> list:
    height, width = image.shape
    data = []
    for y in range(height):
        for x in range(width):
            pixel_value = image[y, x]
            values = [
                pair_id,
                pixel_value,
                x,
                y,
                width,
                height,
            ]
            for dx in range(3):
                for dy in range(3):
                    if dx == 1 and dy == 1:
                        continue
                    x2 = x + dx
                    y2 = y + dy
                    pixel_value2 = 10
                    if x2 >= 0 and x2 < width and y2 >= 0 and y2 < height:
                        pixel_value2 = image[y2, x2]
                    values.append(pixel_value2)
            data.append(values)
    return data

def sample_data(input_data: list, target_data: list, rng) -> list:
    """
    The input_data and target_data can have different lengths. 
    This function makes sure the resulting list have the same length.
    Sample N items from both lists, until all items have been processed.
    """
    # Sample max N times per item.
    input_data_sample_count = np.zeros(len(input_data), dtype=int)
    target_data_sample_count = np.zeros(len(target_data), dtype=int)

    # The unvisited indexes.
    input_data_indexes = np.arange(len(input_data))
    target_data_indexes = np.arange(len(target_data))

    number_of_values_per_sample = 10
    number_of_samples = 400

    input_target_pairs = []
    for i in range(number_of_samples):
        if len(input_data_indexes) < number_of_values_per_sample:
            break

        input_data_sample_indexes = rng.choice(input_data_indexes, number_of_values_per_sample)
        for index in input_data_sample_indexes:
            input_data_sample_count[index] += 1
            if input_data_sample_count[index] == number_of_values_per_sample:
                input_data_indexes = np.delete(input_data_indexes, np.where(input_data_indexes == index))

        # print(f"input_data_sample_indexes: {input_data_sample_indexes}")
        input_data_samples = [input_data[index] for index in input_data_sample_indexes]
        # print(f"input_data_samples: {input_data_samples}")

        if len(target_data_indexes) < number_of_values_per_sample:
            break

        target_data_sample_indexes = rng.choice(target_data_indexes, number_of_values_per_sample)
        for index in target_data_sample_indexes:
            target_data_sample_count[index] += 1
            if target_data_sample_count[index] == number_of_values_per_sample:
                target_data_indexes = np.delete(target_data_indexes, np.where(target_data_indexes == index))
        
        # print(f"target_data_sample_indexes: {target_data_sample_indexes}")
        target_data_samples = [target_data[index] for index in target_data_sample_indexes]
        # print(f"target_data_samples: {target_data_samples}")

        if len(input_data_samples) != len(target_data_samples):
            raise ValueError(f"input and target values have different lengths. input len: {len(input_data_samples)} target len: {len(target_data_samples)}")
        
        input_target_pairs.append((input_data_samples, target_data_samples))
    return input_target_pairs

def count_correct_with_pairs(input_target_pairs: list) -> tuple[int, int]:
    count_correct = 0
    count_total = 0
    for input_data_samples, target_data_samples in input_target_pairs:
        if len(input_data_samples) != len(target_data_samples):
            raise ValueError(f"input and target values have different lengths. input len: {len(input_data_samples)} target len: {len(target_data_samples)}")
        
        n = len(input_data_samples)
        # print(f"n: {n}")
        this_count_correct = 0
        for y in range(n):
            is_target_correct = False
            for x in range(n):
                input_values = input_data_samples[y]
                target_values = target_data_samples[x]

                input_value = input_values[1]
                target_value = target_values[1]

                is_correct = input_value == target_value

                if is_correct:
                    is_target_correct = True
            if is_target_correct:
                this_count_correct += 1
        
        count_correct += (this_count_correct / n)
        count_total += 1
    
    if count_total == 0:
        raise ValueError(f"count_total is zero")

    return count_correct, count_total

class Builder:
    def __init__(self):
        raise NotImplementedError()

    def task_hash(self, value: int):
        raise NotImplementedError()

    def pair_id(self, value: int):
        raise NotImplementedError()

    def position_xy(self, value: int):
        raise NotImplementedError()

    def position_diff(self, value: int):
        raise NotImplementedError()

    def color(self, value: int):
        raise NotImplementedError()

    def size_widthheight(self, value: int):
        raise NotImplementedError()

    def euclidian_distance(self, value: float):
        raise NotImplementedError()

    def unspecified_bool(self, value: bool):
        raise NotImplementedError()

    def build(self):
        raise NotImplementedError()

class BuilderList(Builder):
    def __init__(self):
        self.data = []

    def task_hash(self, value: int):
        pass

    def pair_id(self, value: int):
        self.data.append(value)

    def position_xy(self, value: int):
        self.data.append(value)

    def position_diff(self, value: int):
        self.data.append(value)

    def color(self, value: int):
        self.data.append(value)

    def size_widthheight(self, value: int):
        self.data.append(value)

    def euclidian_distance(self, value: float):
        self.data.append(value)

    def unspecified_bool(self, value: bool):
        int_value = 1 if value else 0
        self.data.append(int_value)

    def build(self):
        return self.data

class VocabularySize:
    TASK_HASH = 256 # 8 bits of the task hash
    PAIR_ID = 31 # allow up to 31 pairs. Usually there are less than 10 pair ids.
    POSITION_XY = 31 # 0-30
    SIZE_WIDTHHEIGHT = 31 # 1-30, 0 is unused
    COLOR = 11 # 0-9 and 10 for unspecified color such as outside the image
    UNSPECIFIED_BOOL = 2 # 0 or 1
    POSITION_DIFF = 61 # -30 to 30
    EUCLIDIAN_DISTANCE = 100 # max distance is ((30 ** 2) * 2) ** 0.5 = 42.426. Stretched to the range 0-99

class Vocabulary:
    TASK_HASH = 0
    PAIR_ID = VocabularySize.TASK_HASH
    POSITION_XY = VocabularySize.TASK_HASH + VocabularySize.PAIR_ID
    SIZE_WIDTHHEIGHT = VocabularySize.TASK_HASH + VocabularySize.PAIR_ID + VocabularySize.POSITION_XY
    COLOR = VocabularySize.TASK_HASH + VocabularySize.PAIR_ID + VocabularySize.POSITION_XY + VocabularySize.SIZE_WIDTHHEIGHT
    UNSPECIFIED_BOOL = VocabularySize.TASK_HASH + VocabularySize.PAIR_ID + VocabularySize.POSITION_XY + VocabularySize.SIZE_WIDTHHEIGHT + VocabularySize.COLOR
    POSITION_DIFF = VocabularySize.TASK_HASH + VocabularySize.PAIR_ID + VocabularySize.POSITION_XY + VocabularySize.SIZE_WIDTHHEIGHT + VocabularySize.COLOR + VocabularySize.UNSPECIFIED_BOOL
    EUCLIDIAN_DISTANCE = VocabularySize.TASK_HASH + VocabularySize.PAIR_ID + VocabularySize.POSITION_XY + VocabularySize.SIZE_WIDTHHEIGHT + VocabularySize.COLOR + VocabularySize.UNSPECIFIED_BOOL + VocabularySize.POSITION_DIFF
    VOCAB_SIZE = VocabularySize.TASK_HASH + VocabularySize.PAIR_ID + VocabularySize.POSITION_XY + VocabularySize.SIZE_WIDTHHEIGHT + VocabularySize.COLOR + VocabularySize.UNSPECIFIED_BOOL + VocabularySize.POSITION_DIFF + VocabularySize.EUCLIDIAN_DISTANCE

class BuilderWithVocabulary(Builder):
    def __init__(self):
        self.data = []

    def task_hash(self, value: int):
        value_int = int(value)
        for _ in range(8):
            b = value_int & 255
            value_int >>= 8
            self.data.append(Vocabulary.TASK_HASH + b)

    def pair_id(self, value: int):
        value_int = int(value)
        if value_int < 0 or value_int >= VocabularySize.PAIR_ID:
            raise ValueError(f"pair_id value out of range: {value}")
        self.data.append(Vocabulary.PAIR_ID + value_int)

    def position_xy(self, value: int):
        value_int = int(value)
        if value_int < 0 or value_int >= VocabularySize.POSITION_XY:
            raise ValueError(f"position_xy value out of range: {value}")
        self.data.append(Vocabulary.POSITION_XY + value_int)

    def position_diff(self, value: int):
        value_int = int(value)
        half = VocabularySize.POSITION_DIFF // 2
        if value_int < -half or value_int >= half:
            raise ValueError(f"position_diff value out of range: {value}")
        self.data.append(Vocabulary.POSITION_DIFF + value_int + half)

    def color(self, value: int):
        value_int = int(value)
        if value_int < 0 or value_int >= VocabularySize.COLOR:
            raise ValueError(f"color value out of range: {value}")
        self.data.append(Vocabulary.COLOR + value_int)

    def size_widthheight(self, value: int):
        value_int = int(value)
        if value_int < 1 or value_int >= VocabularySize.SIZE_WIDTHHEIGHT:
            raise ValueError(f"size_widthheight value out of range: {value}")
        self.data.append(Vocabulary.SIZE_WIDTHHEIGHT + value_int)

    def euclidian_distance(self, value: float):
        value100 = int(remap(value, 1.0, 42.426, 0.0, 99.0))
        if value100 < 0:
            value100 = 0
        if value100 >= VocabularySize.EUCLIDIAN_DISTANCE:
            value100 = VocabularySize.EUCLIDIAN_DISTANCE - 1
        self.data.append(Vocabulary.EUCLIDIAN_DISTANCE + value100)

    def unspecified_bool(self, value: bool):
        value_int = 1 if value else 0
        self.data.append(Vocabulary.UNSPECIFIED_BOOL + value_int)

    def build(self):
        return self.data

def xs_ys_from_input_target_pairs(input_target_pairs: list, task_hash: int, builder_cls) -> tuple[list, list]:
    if issubclass(builder_cls, Builder) == False:
        raise ValueError(f"builder_cls is not a subclass of Builder")
    xs = []
    ys = []
    extra = []
    for input_data_samples, target_data_samples in input_target_pairs:
        if len(input_data_samples) != len(target_data_samples):
            raise ValueError(f"input and target values have different lengths. input len: {len(input_data_samples)} target len: {len(target_data_samples)}")
        
        n = len(input_data_samples)
        # print(f"n: {n}")
        for y in range(n):
            for x in range(n):
                input_values = input_data_samples[y]
                target_values = target_data_samples[x]

                input_pair_index = input_values[DataPoint.PAIR_ID.value]
                input_value = input_values[DataPoint.PIXEL_VALUE.value]
                input_x = input_values[DataPoint.X.value]
                input_y = input_values[DataPoint.Y.value]
                input_width = input_values[DataPoint.WIDTH.value]
                input_height = input_values[DataPoint.HEIGHT.value]
                input_pixel_values2 = input_values[DataPoint.PIXEL_VALUES.value:-1]
                input_x_rev = input_width - input_x - 1
                input_y_rev = input_height - input_y - 1

                target_pair_index = target_values[DataPoint.PAIR_ID.value]
                target_value = target_values[DataPoint.PIXEL_VALUE.value]
                target_x = target_values[DataPoint.X.value]
                target_y = target_values[DataPoint.Y.value]
                target_width = target_values[DataPoint.WIDTH.value]
                target_height = target_values[DataPoint.HEIGHT.value]
                target_x_rev = target_width - target_x - 1
                target_y_rev = target_height - target_y - 1

                dx = input_x - target_x
                dy = input_y - target_y
                distance1 = sqrt(dx * dx + dy * dy)

                same_pair_id_bool = input_pair_index == target_pair_index

                b = builder_cls()
                b.task_hash(task_hash)
                b.position_xy(target_x)
                b.position_xy(target_y)
                b.pair_id(input_pair_index)
                b.pair_id(target_pair_index)
                b.unspecified_bool(same_pair_id_bool)
                b.color(input_value)
                b.position_xy(input_x)
                b.position_xy(input_y)
                b.size_widthheight(input_width)
                b.size_widthheight(input_height)
                b.position_diff(dx)
                b.position_diff(dy)
                b.position_xy(input_x_rev)
                b.position_xy(input_y_rev)
                b.position_xy(target_x_rev)
                b.position_xy(target_y_rev)
                b.euclidian_distance(distance1)
                for j in range(10):
                    # one hot encoding of input_value
                    b.unspecified_bool(input_value == j)
                for pixel_value in input_pixel_values2:
                    b.color(pixel_value)

                xs_item = b.build()

                ys_item = int(target_value)

                extra_item = [
                    input_pair_index,
                    target_pair_index,
                    target_x,
                    target_y,
                ]

                xs.append(xs_item)
                ys.append(ys_item)
                extra.append(extra_item)
    return xs, ys, extra
