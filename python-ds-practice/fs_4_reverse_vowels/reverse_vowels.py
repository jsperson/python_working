def reverse_vowels(s):
    """Reverse vowels in a string.

    Characters which re not vowels do not change position in string, but all
    vowels (y is not a vowel), should reverse their order.

    >>> reverse_vowels("Hello!")
    'Holle!'

    >>> reverse_vowels("Tomatoes")
    'Temotaos'

    >>> reverse_vowels("Reverse Vowels In A String")
    'RivArsI Vewols en e Streng'

    reverse_vowels("aeiou")
    'uoiea'

    reverse_vowels("why try, shy fly?")
    'why try, shy fly?''
    """
    vowels = {"a", "e", "i", "o", "u", "A", "E", "I", "O", "U"}
    vowels_in_string = []
    output = ""
    for x in s:
        if x in vowels:
            vowels_in_string.append(x)
    for x in s:
        if x in vowels:
            output += vowels_in_string.pop(-1)
        else:
            output += x
    return output


print(reverse_vowels("Reverse Vowels In A String"))