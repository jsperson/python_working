import re

fhand = open('./gitrepos/pythonworking/PythonForEverybody/mbox.txt')

lst = list()
lines = list()

for line in fhand:
    #print(line)
    #print(re.findall('*', line))
    #val = re.findall('[A-Z0-9a-z]\S*@\S*[A-Za-z]', line)

    #val = re.findall('^Received:\s(?:from|FROM)\s([A-Za-z0-9.]*) ', line.strip())
    #val = re.findall('(\[[0-9]*\.[0-9]*\.[0-9]*\.[0-9]*\])',line)
    val = re.findall('(\[\d*\.\d*\.\d*\.\d*\])',line)
    if val:
        if val not in lst:
            #lst.append(re.findall('\S+@\S+', line))
            lst.append(val)
            lines.append(line)
    
for element in lst[0:10]:
    for address in element:
        print(address)

for element in range(0,10):
    print(lines[element])