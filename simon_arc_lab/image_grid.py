import random
import numpy as np

class ImageGridBuilder:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.set_separator_size(1)
        self.set_cell_size(1)

    def set_separator_size(self, separator_size: int):
        separator_widths = []
        for _ in range(self.width + 1):
            separator_widths.append(separator_size)
        self.separator_widths = separator_widths

        separator_heights = []
        for _ in range(self.height + 1):
            separator_heights.append(separator_size)
        self.separator_heights = separator_heights

    def set_cell_size(self, cell_size: int):
        cell_widths = []
        for _ in range(self.width):
            cell_widths.append(cell_size)
        self.cell_widths = cell_widths

        cell_heights = []
        for _ in range(self.height):
            cell_heights.append(cell_size)
        self.cell_heights = cell_heights

    def set_cell_size_random(self, seed: int, min_cell_size: int, max_cell_size: int):
        cell_widths = []
        for x in range(self.width):
            cell_width = random.Random(seed + x).randint(min_cell_size, max_cell_size)
            cell_widths.append(cell_width)
        self.cell_widths = cell_widths

        cell_heights = []
        for y in range(self.height):
            cell_height = random.Random(seed + self.width + y).randint(min_cell_size, max_cell_size)
            cell_heights.append(cell_height)
        self.cell_heights = cell_heights

    def set_top_separator_size(self, separator_size: int):
        self.separator_heights[0] = separator_size

    def set_bottom_separator_size(self, separator_size: int):
        self.separator_heights[-1] = separator_size

    def set_left_separator_size(self, separator_size: int):
        self.separator_widths[0] = separator_size

    def set_right_separator_size(self, separator_size: int):
        self.separator_widths[-1] = separator_size

    def draw(self, image: np.array, grid_color: int) -> np.array:
        image_height, image_width = image.shape
        if image_height != self.height or image_width != self.width:
            raise ValueError("The image size does not match the grid size.")
        
        destination_width = sum(self.cell_widths) + sum(self.separator_widths)
        destination_height = sum(self.cell_heights) + sum(self.separator_heights)
        destination_image = np.full((destination_height, destination_width), grid_color, dtype=np.uint8)

        current_y = 0
        for y in range(self.height):
            current_y += self.separator_heights[y]
            if y > 0:
                current_y += self.cell_heights[y-1]
            current_x = 0
            for x in range(self.width):
                current_x += self.separator_widths[x]
                if x > 0:
                    current_x += self.cell_widths[x-1]
                column_width = self.cell_widths[x]
                row_height = self.cell_heights[y]

                for dy in range(row_height):
                    for dx in range(column_width):
                        destination_image[current_y + dy, current_x + dx] = image[y, x]

        return destination_image
