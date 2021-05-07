fhand = open('./gitrepos/pythonworking/PythonForEverybody/words.txt')

dict = {}

for line in fhand:
    line = line.strip().strip('-')
    dict[line] = ''

searchVal = ''

while(searchVal != 'stopit'):
    searchVal = input('Enter value to search for (\"stopit\" to stop): ')
    
    if searchVal == 'stopit': break
    
    if searchVal in dict:
        print('Found ' + searchVal + ' in the file.')
    else:
        print(searchVal + ' not found.')