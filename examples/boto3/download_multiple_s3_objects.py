import boto3
import pandas as pd
import jsp_params as params
s3 = boto3.client('s3', region_name='us-east-1',
                  aws_access_key_id=params.AWS_KEY_ID,
                  aws_secret_access_key=params.AWS_SECRET)

# List to hold dataframes
df_list = []

# Request the list of csv's from S3 using prefix
response = s3.list_objects(
    Bucket='',
    Prefix=''
)

# Get response contents
request_files = response['Contents']

# Iterate over each object
for file in request_files:
    obj = s3.get_object(Bucket='<bucket name', Key=file['Key'])
    # Read it as data frame
    obj_df = pd.read_csv(obj['Body'])
    # Append data fram to list
    df_list.append(obj_df)

# Concatenate all the data frames in the list
df = pd.concat(df_list)

# Preview the dataframe
df.head()
