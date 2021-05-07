# Import create_engine function
from sqlalchemy import create_engine, MetaData, select, Table, func

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
state_fact = Table('state_fact', metadata, autoload=True, autoload_with=engine)
# Build a statement to join census and state_fact tables: stmt
stmt = select([census.columns.pop2000, state_fact.columns.abbreviation])

# Execute the statement and get the first result: result
result = connection.execute(stmt).first()

# Loop over the keys in the result object and print the key and value
for key in result.keys():
    print(key, getattr(result, key))


# Build a statement to select the census and state_fact tables: stmt
stmt = select([census, state_fact])

# Add a select_from clause that wraps a join for the census and state_fact
# tables where the census state column and state_fact name column match
stmt_join = stmt.select_from(
    census.join(state_fact, census.columns.state == state_fact.columns.name))

# Execute the statement and get the first result: result
result = connection.execute(stmt_join).first()

# Loop over the keys in the result object and print the key and value
for key in result.keys():
    print(key, getattr(result, key))

# Build a statement to select the state, sum of 2008 population and census
# division name: stmt
stmt = select([
    census.columns.state,
    func.sum(census.columns.pop2008),
    state_fact.columns.census_division_name
])

# Append select_from to join the census and state_fact tables by the census state and state_fact name columns
stmt_joined = stmt.select_from(
    census.join(state_fact, census.columns.state == state_fact.columns.name)
)

# Append a group by for the state_fact name column
stmt_grouped = stmt_joined.group_by('name','census_division_name')

# Execute the statement and get the results: results
results = connection.execute(stmt_grouped).fetchall()

# Loop over the results object and print each record.
for record in results:
    print(record)


