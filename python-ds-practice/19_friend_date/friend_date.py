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


def friend_date(a, b):
    """Given two friends, do they have any hobbies in common?

    - a: friend #1, a tuple of (name, age, list-of-hobbies)
    - b: same, for friend #2

    Returns True if they have any hobbies in common, False is not.

        >>> elmo = ('Elmo', 5, ['hugging', 'being nice'])
        >>> sauron = ('Sauron', 5000, ['killing hobbits', 'chess'])
        >>> gandalf = ('Gandalf', 10000, ['waving wands', 'chess'])

        >>> friend_date(elmo, sauron)
        False

        >>> friend_date(sauron, gandalf)
        True
    """
    return len(intersection(a[2], b[2])) > 0


if __name__ == "__main__":
    elmo = ("Elmo", 5, ["hugging", "being nice"])
    sauron = ("Sauron", 5000, ["killing hobbits", "chess"])
    gandalf = ("Gandalf", 10000, ["waving wands", "chess"])

    print(friend_date(elmo, sauron))
    print(friend_date(sauron, gandalf))