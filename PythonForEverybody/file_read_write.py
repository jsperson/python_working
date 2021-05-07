rfhand = open("./gitrepos/pythonworking/PythonForEverybody/mbox.txt")

wfhand = open("./gitrepos/pythonworking/PythonForEverybody/scott.txt","w")

total = 0
count = 0

for line in rfhand:
    if line.startswith("X-DSPAM-Confidence:"):
        loc = line.find(":") + 2
        numstr = line[loc:]
        numstr = numstr.strip()
        num = float(numstr)
        total = total + num
        count += 1
        #print(numstr)
        wfhand.write(line)

print("Total:" + str(total))
print("Count:" + str(count))
if(count > 0):
    print("Average:" + str(total/count))

rfhand.close()
wfhand.close()
