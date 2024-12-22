import re
import json
from typing import Union

def digits_from_response(response: str, verbose: bool=False) -> list[list[int]]:
    """
    Extract digits result from a response. Example of response:

    5 6 7
    8 9 0

    It finds the bottom most occurrance of digits and extracts the digits.
    It ignores digits that occurs earlier in the document that are not the last one.
    """

    # Search for all blocks of numbers at the bottom of the response
    lines = response.strip().split('\n')
    bottom_digits = []

    for line in reversed(lines):
        if re.match(r'^\s*(\d+\s*)+$', line):  # Match lines with only digits
            bottom_digits.append(line.strip())
        elif bottom_digits:  # Stop once we've captured the last block of digits
            break

    if not bottom_digits:
        raise ValueError("No digit block found in response.")

    # Reverse to maintain the order of rows
    bottom_digits.reverse()

    # Convert lines of digits into a nested list of integers
    result = [list(map(int, row.split())) for row in bottom_digits]

    if verbose:
        print("Extracted bottom digits:")
        print(result)

    return result