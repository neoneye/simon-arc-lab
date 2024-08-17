import numpy as np

class DecodeRLEError(ValueError):
    """Exception raised for errors in RLE decoding."""
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details

def decode_rle_row_inner(row):
    if not row:
        raise DecodeRLEError("Invalid row: row cannot be empty")

    decoded_row = []
    prev_count = 1
    x = 0

    for ch in row:
        if ch.isdigit():
            color = int(ch)
            for _ in range(prev_count):
                decoded_row.append(color)
                x += 1
            prev_count = 1
        else:
            if not ('a' <= ch <= 'z'):
                raise DecodeRLEError("Invalid character inside row", details=f"Character: {ch}")
            count = ord(ch) - ord('a') + 2
            prev_count = count

    return decoded_row

def decode_rle_row(row, width):
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

def deserialize(input_str):
    verbose = False

    parts = input_str.split(' ')
    count_parts = len(parts)
    if count_parts != 3:
        raise DecodeRLEError("Expected 3 parts", details=f"But got {count_parts} parts")

    width = int(parts[0])
    height = int(parts[1])
    rows = parts[2].split(',')

    count_rows = len(rows)
    if count_rows != height:
        raise DecodeRLEError("Mismatch between height and the number of RLE rows",
                             details=f"Expected height: {height}, Number of rows: {count_rows}")

    image = np.zeros((height, width), dtype=np.uint8)
    copy_y = 0

    for y in range(height):
        row = rows[y]
        if verbose:
            print(f"y: {y} row: {row}")
        if not row:
            if y == 0:
                raise DecodeRLEError("First row is empty")
            image[y, :] = image[copy_y, :]
            continue
        copy_y = y
        decoded_row = decode_rle_row(row, width)
        image[y, :] = decoded_row

    return image
