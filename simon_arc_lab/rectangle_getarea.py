from typing import Tuple
from .rectangle import *

def rectangle_getarea(rect: Rectangle, area_name: str) -> Rectangle:
    def _get_split_a(start: int, size: int) -> Tuple[int, int]:
        size_a = (size + 1) // 2
        return (start, size_a)

    def _get_split_b(start: int, size: int) -> Tuple[int, int]:
        size_b = (size + 1) // 2
        size_a = size - size_b
        start_b = start + size_a
        return (start_b, size_b)

    if area_name == 'top':
        y, height = _get_split_a(rect.y, rect.height)
        return Rectangle(rect.x, y, rect.width, height)
    elif area_name == 'bottom':
        y, height = _get_split_b(rect.y, rect.height)
        return Rectangle(rect.x, y, rect.width, height)
    elif area_name == 'left':
        x, width = _get_split_a(rect.x, rect.width)
        return Rectangle(x, rect.y, width, rect.height)
    elif area_name == 'right':
        x, width = _get_split_b(rect.x, rect.width)
        return Rectangle(x, rect.y, width, rect.height)
    else:
        raise Exception(f"Unknown area_name: {area_name}")
