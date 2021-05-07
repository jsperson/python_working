def multiple_letter_count(phrase):
    """Return dict of {ltr: frequency} from phrase.

    >>> multiple_letter_count('yay')
    {'y': 2, 'a': 1}

    >>> multiple_letter_count('Yay')
    {'Y': 1, 'a': 1, 'y': 1}
    """
    dict = {}
    for l in phrase:
        if l not in dict:
            dict[l] = phrase.count(l)
    return dict


if __name__ == "__main__":
    print(multiple_letter_count("yay"))
    print(multiple_letter_count("Yay"))
