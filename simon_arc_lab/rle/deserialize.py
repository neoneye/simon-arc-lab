import numpy as np

def decode_rle_row_inner(row):
    if not row:
        raise ValueError("invalid row")

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
            raise ValueError("invalid character for full row")

    decoded_row = decode_rle_row_inner(row)
    if len(decoded_row) != width:
        raise ValueError("mismatch width and the number of RLE columns")

    return decoded_row

def deserialize(input_str):
    verbose = False

    parts = input_str.split(' ')
    if len(parts) != 3:
        raise ValueError("invalid input")

    width = int(parts[0])
    height = int(parts[1])
    rows = parts[2].split(',')

    if len(rows) != height:
        raise ValueError("mismatch height and the number of RLE rows")

    image = np.zeros((height, width), dtype=np.uint8)
    copy_y = 0

    for y in range(height):
        row = rows[y]
        if verbose:
            print(f"y: {y} row: {row}")
        if not row:
            if y == 0:
                raise ValueError("first row is empty")
            image[y, :] = image[copy_y, :]
            continue
        copy_y = y
        decoded_row = decode_rle_row(row, width)
        image[y, :] = decoded_row

    return image
