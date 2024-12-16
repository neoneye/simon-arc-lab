import numpy as np
from typing import Optional
from dataclasses import dataclass

@dataclass
class ImageToStringConfig:
    pixel_to_symbol: Optional[dict]
    fallback_symbol: str
    separator_horizontal: str
    separator_vertical: str
    # IDEA: prefix_with_line_number: enum = False|OneBased|ZeroBased
    # IDEA: suffix_with_line_number: enum = False|OneBased|ZeroBased

    def image_to_string(self, image: np.array) -> str:
        height, width = image.shape
        rows = []
        for y in range(height):
            items = []
            for x in range(width):
                pixel = image[y, x]
                if self.pixel_to_symbol is None:
                    symbol = str(pixel)
                elif pixel in self.pixel_to_symbol:
                    symbol = self.pixel_to_symbol[pixel]
                else:
                    symbol = self.fallback_symbol
                items.append(symbol)
            row = self.separator_horizontal.join(items)
            rows.append(row)
        return self.separator_vertical.join(rows)

def image_to_string(image: np.array) -> str:
    """
    Convert an image to a compact string representation, like this:

    from
    [[1, 2, 3], [4, 5, 6]]

    to
    "123\n456"

    Beware that LLM's may group multiple digits into a single token.
    Check how the string representation gets tokenized.
    https://platform.openai.com/tokenizer
    """
    config = ImageToStringConfig(
        pixel_to_symbol=None,
        fallback_symbol='.',
        separator_horizontal='',
        separator_vertical='\n',
    )
    return config.image_to_string(image)

def image_from_string(s: str) -> np.array:
    """
    Convert from a compact string representation to an image, like this:

    from
    "123\n456"

    to
    [[1, 2, 3], [4, 5, 6]]
    """
    rows = s.split("\n")
    height = len(rows)
    max_width = 0
    for y, row in enumerate(rows):
        max_width = max(max_width, len(row))
    width = max_width
    empty_color = 255
    image = np.full((height, width), empty_color, dtype=np.uint8)
    for y, row in enumerate(rows):
        for x, c in enumerate(row):
            if c.isdigit():
                image[y, x] = int(c)
    return image

def image_to_string_long_lowercase_colornames(image: np.array) -> str:
    """
    Convert an image to a long string representation, like this:

    from
    [[1, 2, 3], [4, 5, 6]]

    to
    "blue red green\nyellow grey purple"

    Why use such a long string representation?
    When using LLM's the color names gets translated into 1 token per color.
    Use long color names to reduce the number of tokens.
    https://platform.openai.com/tokenizer
    """
    pixel_to_symbol = {
        0: 'black',
        1: 'blue',
        2: 'red',
        3: 'green',
        4: 'yellow',
        5: 'grey',
        6: 'purple',
        7: 'orange',
        8: 'cyan',
        9: 'brown',
    }
    config = ImageToStringConfig(
        pixel_to_symbol=pixel_to_symbol,
        fallback_symbol='white',
        separator_horizontal=' ',
        separator_vertical='\n',
    )
    return config.image_to_string(image)

