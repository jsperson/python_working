import boto3
import jsp_params as params
s3 = boto3.client('s3', region_name='us-east-1',
                  aws_access_key_id=params.AWS_KEY_ID,
                  aws_secret_access_key=params.AWS_SECRET)

s3.put_object_acl(
    Bucket='',  # Object bucket
    Key='',  # Object name
    ACL='public-read'
)
