def list_scaleup(lst, factor):
    """
    Scale up the list by repeating each element factor times.

    :param lst: The list to scale up, eg. [1, 2, 3]
    :param factor: The factor to scale up by, eg. 2
    :return: The scaled up list. eg. [1, 1, 2, 2, 3, 3]
    """
    return [elem for elem in lst for _ in range(factor)]

def list_compress(lst):
    """
    Compress RLE row, by removing a-z length indicator, and remove duplicate colors adjacent, 
    so it's only the unique pixel colors.

    :param lst: The pixel list to compress, eg. [1, 2, 2, 3, 3, 3]
    :return: The compressed list. eg. [1, 2, 3]
    """
    if not lst:
        return []

    compressed_list = [lst[0]]  # start with the first element

    for i in range(1, len(lst)):
        if lst[i] != lst[i - 1]:
            compressed_list.append(lst[i])
    
    return compressed_list
