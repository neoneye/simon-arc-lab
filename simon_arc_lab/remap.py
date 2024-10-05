def remap(value: float, a: float, b: float, c: float, d: float) -> float:
    """
    Convert value from range [a, b] to range [c, d]
    """
    return c + (d - c) * (value - a) / (b - a)
