def image_size1d_to_string(image_size1d: int) -> str:
    """
    Place typical ARC image size into 3 categories: small, medium, large.
    And sizes that doesn't fit in, use 'other' for that.
    """

    if image_size1d <= 10:
        return 'small'
    if image_size1d <= 20:
        return 'medium'
    if image_size1d <= 30:
        return 'large'
    return 'other'

def histogram_total_to_string(total: int) -> str:
    """
    Convert a histogram sum to a string for use in benchmark.

    My hypothesis is that it's exponential: 
    LLMs struggle adding bigger values.
    and it's easy to add smaller values.
    To test this hypothesis, I will use these categories: a, b, c, d, e.
    And if the value is bigger than 100000, then use 'other'.

    I might need to adjust the categories if the hypothesis is wrong.
    It depends on how skewed the measurements are.
    """

    if total <= 10:
        return 'a'
    if total <= 100:
        return 'b'
    if total <= 1000:
        return 'c'
    if total <= 10000:
        return 'd'
    if total <= 100000:
        return 'e'
    return 'other'
