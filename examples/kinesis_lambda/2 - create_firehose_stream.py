import boto3
import jsp_params as params
firehose = boto3.client('firehose', region_name='us-east-1',
                        aws_access_key_id=params.AWS_KEY_ID,
                        aws_secret_access_key=params.AWS_SECRET)

response = firehose.create_delivery_stream(
    DeliveryStreamName='gps-delivery-stream',
    DeliveryStreamType='DirectPut',
    S3DestinationConfiguration={
        'RoleARN': 'arn:aws:iam::527117955781:role/FirehoseToS3',
        'BucketARN': 'arn:aws:s3:::sd-vehicle-data-jsp'
    }
)

print(response['DeliveryStreamARN'])
