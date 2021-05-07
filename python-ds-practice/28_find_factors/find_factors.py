def find_factors(num):
    """Find factors of num, in increasing order.

    >>> find_factors(10)
    [1, 2, 5, 10]

    >>> find_factors(11)
    [1, 11]

    >>> find_factors(111)
    [1, 3, 37, 111]

    >>> find_factors(321421)
    [1, 293, 1097, 321421]
    """
    # factors = []
    # for i in range(1, num + 1):
    #     if num % i == 0:
    #         factors.append(i)
    # return factors
    n_list = [n for n in range(1, num // 2 + 1) if num % n == 0]

    n_list.append(num)

    return n_list


if __name__ == "__main__":
    print(find_factors(321421))
