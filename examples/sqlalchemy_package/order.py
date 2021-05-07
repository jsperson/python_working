# Import create_engine function
from sqlalchemy import create_engine, MetaData, select, Table

# Create an engine to the census database
#engine = create_engine('postgresql+psycopg2://' + 
#'student:datacamp' + 
#'@postgresql.csrrinzqubik.us-east-1.rds.amazonaws.com' + 
#':5432/census')
import os
file_path = os.path.dirname(os.path.abspath(__file__))
SQLITE_DB = 'census.sqlite'
db_string = 'sqlite:///' + file_path + '/' + SQLITE_DB 
# Create an engine that connects to the census.sqlite file: engine
engine = create_engine(db_string)
# Create a metadata object: metadata
metadata = MetaData()
# Create a connection on engine
connection = engine.connect()
# Reflect census table via engine: census
census = Table('census', metadata, autoload=True, autoload_with=engine)
# Create a select query: stmt
# Build a query to select the state column: stmt
stmt = select([census.columns.state])

# Order stmt by the state column
stmt = stmt.order_by(census.columns.state)

# Execute the query and store the results: results
results = connection.execute(stmt).fetchall()

# Print the first 10 results
print(results[:10])

# Import desc
from sqlalchemy import desc

# Build a query to select the state column: stmt
stmt = select([census.columns.state, census.columns.sex])

# Order stmt by state in descending order: rev_stmt
rev_stmt = stmt.order_by(desc(census.columns.state), census.columns.sex)

# Execute the query and store the results: rev_results
rev_results = connection.execute(rev_stmt).fetchall()

# Print the first 10 rev_results
print(rev_results[:10])

# Build a query to select state and age: stmt
stmt = select([census.columns.state, census.columns.age])

# Append order by to ascend by state and descend by age
stmt = stmt.order_by(census.columns.state, desc(census.columns.age))

# Execute the statement and store all the records: results
results = connection.execute(stmt).fetchall()

# Print the first 20 results
print(results[:20])
