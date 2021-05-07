def repeat(phrase, num):
    """Return phrase, repeated num times.

        >>> repeat('*', 3)
        '***'

        >>> repeat('abc', 2)
        'abcabc'

        >>> repeat('abc', 0)
        ''

    Ignore illegal values of num and return None:

        >>> repeat('abc', -1) is None
        True

        >>> repeat('abc', 'nope') is None
        True
    """
    try:
        if num < 0:
            return None
        return phrase * num
    except:
        return None


print(repeat("abc", "nope") is None)
print(repeat("abc", 2))
print(repeat("abc", -1))
