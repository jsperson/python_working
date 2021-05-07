def flip_case(phrase, to_swap):
    """Flip [to_swap] case each time it appears in phrase.

    >>> flip_case('Aaaahhh', 'a')
    'aAAAhhh'

    >>> flip_case('Aaaahhh', 'A')
    'aAAAhhh'

    >>> flip_case('Aaaahhh', 'h')
    'AaaaHHH'

    """
    to_swap = to_swap.lower()
    new_phrase = ""
    for letter in phrase:
        if letter.lower() == to_swap:
            letter = letter.swapcase()
        new_phrase += letter
    return new_phrase


if __name__ == "__main__":
    print(flip_case("Aaaahhh", "h"))
