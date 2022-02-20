import boto3
import pandas as pd
import jsp_params as params
s3 = boto3.client('s3', region_name='us-east-1',
                  aws_access_key_id=params.AWS_KEY_ID,
                  aws_secret_access_key=params.AWS_SECRET)

# Option 1 - download entire file
s3.download_file(
    Filename='potholes_local.csv',  # filename to land
    Bucket='',
    Key=''
)

# Then can use pandas to read from disk
df = pd.read_csv('<filename>')

# Option 2 - use get_object to stream file
obj = s3.get_object(
    Bucket='',
    Key=''
)
df = pd.read_csv(obj['Body'])

# Option 3 - presigned URLs
share_url = s3.generate_presigned_url(
    ClientMethod='get_object',
    ExpiresIn=3600,
    Params={'Bucket': '<bucket name>', 'Key': '<object key'}
)
df = pd.read_csv(share_url)
