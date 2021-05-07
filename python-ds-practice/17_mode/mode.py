def mode(nums):
    """Return most-common number in list.

    For this function, there will always be a single-most-common value;
    you do not need to worry about handling cases where more than one item
    occurs the same number of times.

        >>> mode([1, 2, 1])
        1

        >>> mode([2, 2, 3, 3, 2])
        2
    """
    counter = {}
    for x in nums:
        if x in counter:
            counter[x] += 1
        else:
            counter[x] = 0

    max_count = max(counter.values())

    for k, v in counter.items():
        if v == max_count:
            return k


if __name__ == "__main__":
    print(mode([2, 2, 3, 3, 2]))