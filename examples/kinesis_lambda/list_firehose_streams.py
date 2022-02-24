import boto3
import jsp_params as params
firehose = boto3.client('firehose', region_name='us-east-1',
                        aws_access_key_id=params.AWS_KEY_ID,
                        aws_secret_access_key=params.AWS_SECRET)

response = firehose.list_delivery_streams()
print(response['DeliveryStreamNames'])

# Delete all streams - DANGER! Block quoted for safety
'''for stream_name in response['DeliveryStreamNames']:
    firehose.delete_delivery_stream(DeliveryStreamName=stream_name)'''
