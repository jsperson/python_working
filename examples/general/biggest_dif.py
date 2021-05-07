import sys

chars = sys.argv[1:]

nums = list(map(int, chars))

smallest = min(nums)
largest = max(nums)

print(str(smallest) + ' ' + str(largest) + ' ' + str(abs(smallest - largest)))
