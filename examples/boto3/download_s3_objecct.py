import boto3
import jsp_params as params
s3 = boto3.client('s3', region_name='us-east-1',
                  aws_access_key_id=params.AWS_KEY_ID,
                  aws_secret_access_key=params.AWS_SECRET)

response = s3.download_file(
    Filename='list_s3.py.tmp',
    Bucket='scott-test-bucket-1972',
    Key='list_s3.py'
)
