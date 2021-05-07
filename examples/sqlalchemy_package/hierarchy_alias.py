# Import create_engine function
from sqlalchemy import create_engine, MetaData, select, Table, func

# Create an engine to the census database
#engine = create_engine('postgresql+psycopg2://' + 
#'student:datacamp' + 
#'@postgresql.csrrinzqubik.us-east-1.rds.amazonaws.com' + 
#':5432/census')
import os
file_path = os.path.dirname(os.path.abspath(__file__))
SQLITE_DB = 'employees.sqlite'
db_string = 'sqlite:///' + file_path + '/' + SQLITE_DB 
# Create an engine that connects to the census.sqlite file: engine
engine = create_engine(db_string)
# Create a metadata object: metadata
metadata = MetaData()
# Create a connection on engine
connection = engine.connect()
# Reflect census table via engine: census
employees = Table('employees', metadata, autoload=True, autoload_with=engine)

# Make an alias of the employees table: managers
managers = employees.alias()

# Build a query to select names of managers and their employees: stmt
stmt = select(
    [managers.columns.name.label('manager'),
     employees.columns.name.label('employee')]
)

# Match managers id with employees mgr: stmt_matched
stmt_matched = stmt.where(managers.columns.id == employees.columns.mgr)

# Order the statement by the managers name: stmt_ordered
stmt_ordered = stmt_matched.order_by('manager')

# Execute statement: results
results = connection.execute(stmt_ordered).fetchall()

# Print records
for record in results:
    print(record)

# Make an alias of the employees table: managers
managers = employees.alias()

# Build a query to select names of managers and counts of their employees: stmt
stmt = select([managers.columns.name, func.count(employees.columns.id)])

# Append a where clause that ensures the manager id and employee mgr are equal
stmt_matched = stmt.where(managers.columns.id == employees.columns.mgr)

# Group by Managers Name
stmt_grouped = stmt_matched.group_by(managers.columns.name)

# Execute statement: results
results = connection.execute(stmt_grouped).fetchall()

# print manager
for record in results:
    print(record)