import boto3
import jsp_params as params
firehose = boto3.client('firehose', region_name='us-east-1',
                        aws_access_key_id=params.AWS_KEY_ID,
                        aws_secret_access_key=params.AWS_SECRET)
