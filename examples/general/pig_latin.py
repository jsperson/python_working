import sys

pig_latin = ''
word = ''

sentence = sys.argv[1:]

for i in sentence:
    word = i[1:] + i[0] + 'ay' + ' '
    pig_latin += word
print(pig_latin)
