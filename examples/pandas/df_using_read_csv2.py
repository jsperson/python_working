import pandas as pd
import os

file_path = os.path.dirname(os.path.abspath(__file__))
file_messy = f'{file_path}/messy_stock_data.tsv'
file_clean = f'{file_path}/tmp_clean_stock_data.csv'
# Read the raw file as-is: df1
df1 = pd.read_csv(file_messy)

# Print the output of df1.head()
print(df1.head(5))

# Read in the file with the correct parameters: df2
df2 = pd.read_csv(file_messy, delimiter=' ', header=3, comment='#')

# Print the output of df2.head()
print(df2.head())

# Save the cleaned up DataFrame to a CSV file without the index
df2.to_csv(file_clean, index=False)

# Save the cleaned up DataFrame to an excel file without the index
df2.to_excel(f'{file_path}/file_clean.xlsx', index=False)
