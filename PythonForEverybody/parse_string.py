str = 'X-DSPAM-Confidence: 0.8475'
loc = str.find(":")
num = str[loc + 2:]
num = float(num)
print("The extracted number is %g." % num)