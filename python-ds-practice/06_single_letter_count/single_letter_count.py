def single_letter_count(word, letter):
    """How many times does letter appear in word (case-insensitively)?

    >>> single_letter_count('Hello World', 'h')
    1

    >>> single_letter_count('Hello World', 'z')
    0

    >>> single_letter_count("Hello World", 'l')
    3
    Best answer: use the list.count() function.
    return word.lower().count(letter.lower())
    """
    word = word.lower()
    letter = letter.lower()
    count = 0

    for l in word:
        if l == letter:
            count += 1
    return count


if __name__ == "__main__":
    print(single_letter_count("EllieJason", "e"))
