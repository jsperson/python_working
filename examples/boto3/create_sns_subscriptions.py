import boto3
import jsp_params as params
sns = boto3.client('sns', region_name='us-east-1',
                   aws_access_key_id=params.AWS_KEY_ID,
                   aws_secret_access_key=params.AWS_SECRET)

response = sns.subscribe(
    TopicArn='',
    Protocol='SMS',
    Endpoint='7038673476'
)

response = sns.subscribe(
    TopicArn='',
    Protocol='email',
    Endpoint='jsperson@nospam.com'
)
