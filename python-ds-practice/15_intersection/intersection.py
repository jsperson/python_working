def intersection(l1, l2):
    """Return intersection of two lists as a new list::

    >>> intersection([1, 2, 3], [2, 3, 4])
    [2, 3]

    >>> intersection([1, 2, 3], [1, 2, 3, 4])
    [1, 2, 3]

    >>> intersection([1, 2, 3], [3, 4])
    [3]

    >>> intersection([1, 2, 3], [4, 5, 6])
    []
    """
    return [li for li in l1 if li in l2]


if __name__ == "__main__":
    print(intersection([1, 2, 3], [3, 4]))
    print(intersection([1, 2, 3], [4, 5, 6]))