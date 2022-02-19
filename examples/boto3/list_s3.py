import boto3
import jsp_params as params
s3 = boto3.client('s3', region_name='us-east-1',
                  aws_access_key_id=params.AWS_KEY_ID,
                  aws_secret_access_key=params.AWS_SECRET)

response = s3.list_buckets()
# Iterate over Buckets from .list_buckets() response
for bucket in response['Buckets']:

    # Print the Name for each bucket
    print(bucket['Name'])
