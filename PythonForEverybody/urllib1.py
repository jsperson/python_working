import urllib.request

fhand = urllib.request.urlopen('http://data.pr4e.org/romeo.txt')

wordCount = dict()

for line in fhand:
    words = line.decode().split()
    for word in words:
        wordCount[word] = wordCount.get(word, 0) + 1
        
print(wordCount)
    