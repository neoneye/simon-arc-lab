import numpy as np
from typing import Optional
from dataclasses import dataclass

@dataclass
class ImageToString:
    pixel_to_symbol: Optional[dict]
    fallback_symbol: str
    separator_horizontal: str
    separator_vertical: str
    top_column_mode: Optional[str] # None|'A'
    bottom_column_mode: Optional[str] # None|'A'
    prefix_column_symbol: Optional[str] # None|''|' '
    prefix_with_line_number: Optional[str] # None|'1'|'-1'
    # IDEA: suffix_with_line_number: enum = False|OneBased|ZeroBased

    @staticmethod
    def spreadsheet_column_name(column_index: int) -> str:
        if column_index < 26:
            # A-Z
            return chr(ord('A') + column_index)
        else:
            # 'AA', 'AB', 'AC', ... 'ZZ'
            return chr(ord('A') + column_index // 26 - 1) + chr(ord('A') + column_index % 26)

    def convert_pixel_to_symbol(self, pixel_value: int) -> str:
        if self.pixel_to_symbol is None:
            return str(pixel_value)
        if pixel_value in self.pixel_to_symbol:
            return self.pixel_to_symbol[pixel_value]
        return self.fallback_symbol

    def apply(self, image: np.array) -> str:
        height, width = image.shape
        rows = []

        def append_column_names(column_name_mode: Optional[str]):
            if column_name_mode is None:
                return
            if column_name_mode == 'A':
                items = []
                if self.prefix_column_symbol is not None:
                    items.append(self.prefix_column_symbol)
                for x in range(width):
                    items.append(self.spreadsheet_column_name(x))
                row = self.separator_horizontal.join(items)
                rows.append(row)
                return
            raise ValueError(f"Invalid column_name_mode: {column_name_mode}")

        # top row with column names
        append_column_names(self.top_column_mode)

        # loop over the rows
        for y in range(height):
            items = []
            if self.prefix_with_line_number == '1':
                items.append(str(y + 1))
            if self.prefix_with_line_number == '-1':
                items.append(str(height - y))
            
            # loop over the pixels in the current row
            for x in range(width):
                pixel_value = image[y, x]
                symbol = self.convert_pixel_to_symbol(pixel_value)
                items.append(symbol)
            row = self.separator_horizontal.join(items)
            rows.append(row)

        # bottom row with column names
        append_column_names(self.bottom_column_mode)
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
    config = ImageToString(
        pixel_to_symbol=None,
        fallback_symbol='.',
        separator_horizontal='',
        separator_vertical='\n',
        top_column_mode=None,
        bottom_column_mode=None,
        prefix_column_symbol=None,
        prefix_with_line_number=None,
    )
    return config.apply(image)

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

# ARC-AGI color names

COLORNAME_BLACK = 'black'
COLORNAME_BLUE = 'blue'
COLORNAME_RED = 'red'
COLORNAME_GREEN = 'green'
COLORNAME_YELLOW = 'yellow'
COLORNAME_GREY = 'grey'
COLORNAME_PURPLE = 'purple'
COLORNAME_ORANGE = 'orange'
COLORNAME_CYAN = 'cyan'
COLORNAME_BROWN = 'brown'
COLORNAME_WHITE = 'white'

COLORID_TO_COLORNAME = {
    0: COLORNAME_BLACK,
    1: COLORNAME_BLUE,
    2: COLORNAME_RED,
    3: COLORNAME_GREEN,
    4: COLORNAME_YELLOW,
    5: COLORNAME_GREY,
    6: COLORNAME_PURPLE,
    7: COLORNAME_ORANGE,
    8: COLORNAME_CYAN,
    9: COLORNAME_BROWN,
}

IMAGETOSTRING_COLORNAME = ImageToString(
    pixel_to_symbol=COLORID_TO_COLORNAME,
    fallback_symbol=COLORNAME_WHITE,
    separator_horizontal=' ',
    separator_vertical='\n',
    top_column_mode=None,
    bottom_column_mode=None,
    prefix_column_symbol=None,
    prefix_with_line_number=None,
)

def image_to_string_colorname(image: np.array) -> str:
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
    return IMAGETOSTRING_COLORNAME.apply(image)

def image_to_string_spreadsheet_v1(image: np.array) -> str:
    """
    Convert an image to a spreadsheet like string representation, like this:

    from
    [[1, 2, 3], [4, 5, 6]]

    to
    ",A,B,C\n1,1,2,3\n2,4,5,6"

    Why use an spreadsheet like string representation?
    Spreadsheets are used everywhere, and LLM's may have been trained on spreadsheets.
    Maybe it's a good representation for LLM's.
    """
    config = ImageToString(
        pixel_to_symbol=None,
        fallback_symbol='error',
        separator_horizontal=',',
        separator_vertical='\n',
        top_column_mode='A',
        bottom_column_mode=None,
        prefix_column_symbol='',
        prefix_with_line_number='1',
    )
    return config.apply(image)

COLORID_TO_EMOJI_CIRCLE_V1 = {
    0: '⚫',  # black circle
    1: '🔵',  # blue circle
    2: '🔴',  # red circle
    3: '🟢',  # green circle
    4: '🟡',  # yellow circle
    5: '⚪',  # white circle (used as grey)
    6: '🟣',  # purple circle
    7: '🟠',  # orange circle
    8: '🟦',  # blue square (used for cyan)
    9: '🟤',  # brown circle
}

IMAGETOSTRING_EMOJI_CIRCLE_V1 = ImageToString(
    pixel_to_symbol=COLORID_TO_EMOJI_CIRCLE_V1,
    fallback_symbol='❌',
    separator_horizontal='',
    separator_vertical='\n',
    top_column_mode=None,
    bottom_column_mode=None,
    prefix_column_symbol=None,
    prefix_with_line_number=None,
)

def image_to_string_emoji_circles_v1(image: np.array) -> str:
    """
    Convert an image to an emoji string representation, like this:

    from
    [[1, 2, 3], [4, 5, 6]]

    to
    "🔵🔴🟢\n🟡⚪🟣"

    Why use an emoji string representation?
    The emoji's gets tokenized with variable length between 1 and 3 tokens for each emoji.
    It doesn't tokenize well, I doubt that it's a good representation for LLM's.
    https://platform.openai.com/tokenizer
    """
    return IMAGETOSTRING_EMOJI_CIRCLE_V1.apply(image)

def image_to_string_emoji_chess_without_indices_v1(image: np.array) -> str:
    """
    Convert an image to an chess string representation, like this:

    from
    [[1, 2, 3], [4, 5, 6]]

    to
    "♕♖♗\n♘♙♚"

    Why use an chess string representation?
    The chess game have their own unicode symbols.
    Maybe it's a good representation for LLM's.
    The chess emoji's gets tokenized with variable length between 1 and 3 tokens for each chess emoji.
    It doesn't tokenize well, I doubt that it's a good representation for LLM's.
    https://platform.openai.com/tokenizer
    """
    pixel_to_symbol = {
        0: '♔', # black
        1: '♕', # blue
        2: '♖', # red
        3: '♗', # green
        4: '♘', # yellow
        5: '♙', # grey
        6: '♚', # purple
        7: '♛', # orange
        8: '♜', # cyan
        9: '♝', # brown
    }
    config = ImageToString(
        pixel_to_symbol=pixel_to_symbol,
        fallback_symbol='♞',
        separator_horizontal='',
        separator_vertical='\n',
        top_column_mode=None,
        bottom_column_mode=None,
        prefix_column_symbol=None,
        prefix_with_line_number=None,
    )
    return config.apply(image)

def image_to_string_emoji_chess_with_indices_v1(image: np.array) -> str:
    """
    Convert an image to an chess string representation, like this:

    from
    [[1, 2, 3], [4, 5, 6]]

    to
    "2♕♖♗\n1♘♙♚\n ABC"

    Why use an chess string representation?
    The chess game have their own unicode symbols.
    Maybe it's a good representation for LLM's.
    The chess emoji's gets tokenized with variable length between 1 and 3 tokens for each chess emoji.
    It doesn't tokenize well, I doubt that it's a good representation for LLM's.
    https://platform.openai.com/tokenizer
    """
    pixel_to_symbol = {
        0: '♔', # black
        1: '♕', # blue
        2: '♖', # red
        3: '♗', # green
        4: '♘', # yellow
        5: '♙', # grey
        6: '♚', # purple
        7: '♛', # orange
        8: '♜', # cyan
        9: '♝', # brown
    }
    config = ImageToString(
        pixel_to_symbol=pixel_to_symbol,
        fallback_symbol='♞',
        separator_horizontal='',
        separator_vertical='\n',
        top_column_mode=None,
        bottom_column_mode='A',
        prefix_column_symbol=' ',
        prefix_with_line_number='-1',
    )
    return config.apply(image)
