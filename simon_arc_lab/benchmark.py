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
