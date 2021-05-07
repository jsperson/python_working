"""Print the count of each word in a string ignoring consecutive identical words"""

sentence = 'this this this is a sample this sample'
word_list = sentence.split()

word_count = {}
last = ''

for i in range(len(word_list)):
    if word_list[i] == last:
        continue
    else:
        last = word_list[i]
    if last in word_count:
        word_count[last] += 1
    else:
        word_count[last] = 1
print(word_count)

# Sort the output dictionary
# https://www.askpython.com/python/dictionary/sort-a-dictionary-in-python#:~:text=%20How%20to%20Sort%20a%20Dictionary%20in%20Python%3F,both%20by%20key...%204%20References.%20%20More%20
#my_dict = {1: 2, 2: 10, "Hello": 1234}
#print({key: value for key, value in sorted(my_dict.items(), key=lambda item: item[0])})
print({key: value for key, value in sorted(
    word_count.items(), key=lambda item: item[0])})
