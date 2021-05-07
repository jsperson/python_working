# Import package
from urllib.request import urlretrieve
import os

file_path = os.path.dirname(os.path.abspath(__file__)) + "/"

# Import pandas
import pandas as pd

# Assign url of file: url
url = "https://s3.amazonaws.com/assets.datacamp.com/production/course_1606/datasets/winequality-red.csv"

# Save file locally
urlretrieve(url, file_path + "winequality-red.csv")

# Read file into a DataFrame and print its head
df = pd.read_csv(file_path + "winequality-red.csv", sep=";")
print(df.head())