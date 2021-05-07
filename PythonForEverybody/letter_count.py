fhand = open('./gitrepos/pythonworking/PythonForEverybody/words.txt')

#dictionary = dict()
counts = dict()

for line in fhand:
    line = line.upper().strip().strip('-')
    for char in line:
        if char not in counts: counts[char] = 1
        else: counts[char] += 1

s = [(k, counts[k]) for k in sorted(counts, key=counts.get, reverse=True)]

for k, v in s:
    print(k,v)
    