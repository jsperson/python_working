def list_check(lst):
    """Are all items in lst a list?

    >>> list_check([[1], [2, 3]])
    True

    >>> list_check([[1], "nope"])
    False
    """
    for x in lst:
        if not (isinstance(x, list)):
            return False
    return True


if __name__ == "__main__":
    print(list_check([[1], [2, 3]]))
    print(list_check([[1], "nope"]))