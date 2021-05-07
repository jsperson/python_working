import os
from sqlalchemy import create_engine, MetaData, Table
file_path = os.path.dirname(os.path.abspath(__file__))
SQLITE_DB = 'census.sqlite'
db_string = 'sqlite:///' + file_path + '/' + SQLITE_DB 

# Create an engine that connects to the census.sqlite file: engine
engine = create_engine(db_string)

# Create a metadata object: metadata
metadata = MetaData()

# Reflect census table from the engine: census
census = Table('census', metadata, autoload=True, autoload_with=engine)

# Print census table metadata
print(repr(census))

# Print the column names
print(census.columns.keys())

