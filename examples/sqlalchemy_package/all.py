# Import create_engine, MetaData
from sqlalchemy import create_engine, MetaData
import os
file_path = os.path.dirname(os.path.abspath(__file__))
SQLITE_DB = 'chapter5.sqlite'
db_string = 'sqlite:///' + file_path + '/' + SQLITE_DB 
# Create an engine that connects to the census.sqlite file: engine
engine = create_engine(db_string)

# Initialize MetaData: metadata
metadata = MetaData()
# Import Table, Column, String, and Integer
from sqlalchemy import Table, Column, String, Integer

# Build a census table: census
census = Table('census', metadata,
               Column('state', String(30)),
               Column('sex', String(1)),
               Column('age', Integer),
               Column('pop2000', Integer),
               Column('pop2008', Integer))

# Create the table in the database
metadata.create_all(engine)

# import pandas
import pandas as pd

# read census.csv into a DataFrame : census_df
census_df = pd.read_csv('census.csv', header=None)

# Create an empty list: values_list
values_list = []

# Iterate over the rows
for row in csv_reader:
    # Create a dictionary with the values
    data = {'state': row[0], 'sex': row[1], 'age':row[2], 'pop2000': row[3],
            'pop2008': row[4]}
    # Append the dictionary to the values list
    values_list.append(data)

# Import insert
from sqlalchemy import insert

# Build insert statement: stmt
stmt = insert(census)

# Use values_list to insert data: results
results = connection.execute(stmt, values_list)

# Print rowcount
print(results.rowcount)
