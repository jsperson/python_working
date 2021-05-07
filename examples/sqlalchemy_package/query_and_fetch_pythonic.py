import os
from sqlalchemy import create_engine, Table, select, MetaData
file_path = os.path.dirname(os.path.abspath(__file__))
SQLITE_DB = 'census.sqlite'
db_string = 'sqlite:///' + file_path + '/' + SQLITE_DB 

engine = create_engine(db_string)
# Create a metadata object: metadata
metadata = MetaData()
# Create a connection on engine
connection = engine.connect()
# Reflect census table via engine: census
census = Table('census', metadata, autoload=True, autoload_with=engine)
# Build select statement for census table: stmt
stmt = select([census])
# Print the emitted statement to see the SQL string
print(stmt)
# Execute the statement on connection and fetch 10 records: result
results = connection.execute(stmt).fetchmany(size=10)

# Execute the statement and print the results
print(results)

# Get the first row of the results by using an index: first_row
first_row = results[0]

# Print the first row of the results
print(first_row)

# Print the first column of the first row by accessing it by its index
print(first_row[0])

# Print the 'state' column of the first row by using its name
print(first_row['state'])