# Import create_engine function
from sqlalchemy import create_engine

# Create an engine to the census database
engine = create_engine('mysql+pymysql://' + #(the dialect and driver).
'student:datacamp' + #(the username and password).
'@courses.csrrinzqubik.us-east-1.rds.amazonaws.com:3306/' + #(the host and port).
'census')

# Print the table names
print(engine.table_names())