from typing import List, Optional
import numpy as np
from ..remap import remap

class DecodeRLEError(ValueError):
    """Exception raised for errors in RLE decoding."""
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message)
        self.details = details

class DeserializeError(ValueError):
    """Exception raised for errors in deserialize."""
    def __init__(self, message: str, score: float, details: Optional[str] = None, decode_rle_error: Optional[DecodeRLEError] = None):
        super().__init__(message)
        self.details = details
        self.score = score
        self.decode_rle_error = decode_rle_error

def decode_rle_row_inner(row: str) -> List[int]:
    if not row:
        raise DecodeRLEError("Invalid row: row cannot be empty")

    decoded_row = []
    prev_count = 1
    x = 0
    current_az_count = 0

    for ch in row:
        if ch.isdigit():
            color = int(ch)
            for _ in range(prev_count):
                decoded_row.append(color)
                x += 1
            prev_count = 1
            current_az_count = 0
        else:
            if not ('a' <= ch <= 'z'):
                raise DecodeRLEError("Invalid character inside row", details=f"Character: {ch}")
            current_az_count += 1
            if current_az_count >= 2:
                raise DecodeRLEError("No adjacent a-z characters are allowed", details=f"Character: {ch}")
            count = ord(ch) - ord('a') + 2
            prev_count = count

    if current_az_count > 0:
        raise DecodeRLEError("Last character must not be a-z character", details=f"Character: {ch}")

    return decoded_row

def decode_rle_row(row: str, width: int) -> List[int]:
    if not row:
        return []

    if len(row) == 1:
        ch = row[0]
        if ch.isdigit():
            color = int(ch)
            return [color] * width
        else:
            raise DecodeRLEError("Invalid character for full row", details=f"Character: {ch}")

    decoded_row = decode_rle_row_inner(row)
    length_decoded_row = len(decoded_row)
    if length_decoded_row != width:
        raise DecodeRLEError("Mismatch between width and the number of RLE columns",
                             details=f"Expected width: {width}, Decoded width: {length_decoded_row}")

    return decoded_row

def deserialize(input_str: str) -> np.array:
    verbose = False

    parts = input_str.split(' ')
    count_parts = len(parts)
    if count_parts != 3:
        raise DeserializeError("Expected 3 parts", score=0, details=f"But got {count_parts} parts")

    width_str, height_str, rows_str = parts
    rows = rows_str.split(',')

    # Validate the "width" string
    try:
        width = int(width_str)
    except ValueError as e:
        raise DeserializeError(
            "Cannot parse width",
            score=1,
            details=str(e)
        )

    # Images with negative dimensions cannot be created
    if width < 0:
        raise DeserializeError(
            "Width must non-negative",
            score=2
        )

    # Validate the "height" string
    try:
        height = int(height_str)
    except ValueError as e:
        raise DeserializeError(
            "Cannot parse height",
            score=3,
            details=str(e)
        )

    # Images with negative dimensions cannot be created
    if height < 0:
        raise DeserializeError(
            "Height must non-negative",
            score=4
        )

    count_rows = len(rows)
    if count_rows != height:
        raise DeserializeError(
            "Mismatch between height and the number of RLE rows",
            score=5,
            details=f"Expected height: {height}, Number of rows: {count_rows}"
        )

    image = np.zeros((height, width), dtype=np.uint8)
    copy_y = 0
    count_valid_row = 0
    for y in range(height):
        row = rows[y]
        if verbose:
            print(f"y: {y} row: {row}")
        if not row:
            if y == 0:
                raise DeserializeError(
                    "First row is empty",
                    score=6
                )
            image[y, :] = image[copy_y, :]
            continue
        copy_y = y
        try:
            decoded_row = decode_rle_row(row, width)
            image[y, :] = decoded_row
            count_valid_row += 1
        except DecodeRLEError as e:
            min_score = 7
            max_score = 99
            if height < 2:
                score = min_score
            else:
                score = remap(count_valid_row, 0, height-1, min_score, max_score)
            raise DeserializeError(
                "Cannot deserialize row",
                score=score,
                details=f"y: {y} height: {height}",
                decode_rle_error=e
            )

    return image
