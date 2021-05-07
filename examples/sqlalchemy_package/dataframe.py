# Import create_engine function
from sqlalchemy import create_engine, MetaData, select, Table, func
# import pandas
import pandas as pd
import matplotlib.pyplot as plt

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

# Build an expression to calculate the sum of pop2008 labeled as population
pop2008_sum = func.sum(census.columns.pop2008).label('population')

# Build a query to select the state and sum of pop2008: stmt
stmt = select([census.columns.state, pop2008_sum])

# Group stmt by state
stmt = stmt.group_by(census.columns.state)

# Execute the statement and store all the records: results
results = connection.execute(stmt).fetchall()

# Print results
print(results)

# Create a DataFrame from the results: df
df = pd.DataFrame(results)

# Set column names
df.columns = results[0].keys()

# Print the DataFrame
print(df)

# Plot the DataFrame
df.plot.bar()
plt.show()
