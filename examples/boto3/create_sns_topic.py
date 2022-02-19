import boto3
import jsp_params as params
sns = boto3.client('sns', region_name='us-east-1',
                   aws_access_key_id=params.AWS_KEY_ID,
                   aws_secret_access_key=params.AWS_SECRET)

response = sns.create_topic(Name='scott_person')
topic_arn = response['TopicArn']
print(topic_arn)

response = sns.delete_topic(TopicArn=topic_arn)
