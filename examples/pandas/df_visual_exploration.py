import matplotlib.pyplot as plt
import pandas as pd
import os

file_path = os.path.dirname(os.path.abspath(__file__))

file_messy = f'{file_path}/messy_stock_data.tsv'

# Read in the file with the correct parameters: df2
df = pd.read_csv(file_messy, delimiter=' ', header=3, comment='#')

print(df.head())

df = df.transpose()
new_header = df.iloc[0]  # grab the first row for the header
df = df[1:]  # take the data less the header row
df.columns = new_header  # set the header row as the df header

df.reset_index(inplace=True)

df.rename(columns={'index': 'Month'}, inplace=True)

print(df.head())

# Create a list of y-axis column names: y_columns
y_columns = ['APPLE', 'IBM']

# Generate a line plot
df.plot(x='Month', y=y_columns)

# Add the title
plt.title('Monthly stock prices')

# Add the y-axis label
plt.ylabel('Price ($US)')

# Display the plot
plt.show()
