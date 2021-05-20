import pandas as pd

list_keys = ['Country', 'Total']
list_values = [['United States', 'Soviet Union',
                'United Kingdom'], [1118, 473, 273]]

# Zip the 2 lists together into one list of (key,value) tuples: zipped
zipped = list(zip(list_keys, list_values))

# Inspect the list using print()
print(zipped)

# Build a dictionary with the zipped list: data
data = dict(zipped)

# Build and inspect a DataFrame from the dictionary: df
df = pd.DataFrame(data)

# Change column names
df.columns = ['Country', 'Total Medals']
print(df)
