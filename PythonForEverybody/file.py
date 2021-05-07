fhand = open("./gitrepos/pythonworking/PythonForEverybody/mbox.txt")
count = 0
string = ''

for line in fhand:
    count += 1
    string = string + line
    
    if line.startswith("From:"): print(line.strip())

fhand.close()

print("Line Count: %d " % count)
print ("Length: %d" % len(string))
