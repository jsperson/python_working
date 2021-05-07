from sqlalchemy import create_engine, inspect
import os
file_path = os.path.dirname(os.path.abspath(__file__))
SQLITE_DB = 'census.sqlite'
db_string = 'sqlite:///' + file_path + '/' + SQLITE_DB 

# Create an engine that connects to the census.sqlite file: engine
engine = create_engine(db_string)
insp = inspect(engine)
print(insp.get_table_names())

# Print table names - this method has been depractated
#print(engine.table_names())