import boto3
import jsp_params as params
sns = boto3.client('sns', region_name='us-east-1',
                   aws_access_key_id=params.AWS_KEY_ID,
                   aws_secret_access_key=params.AWS_SECRET)

response = sns.unsubscribe(
    SubscriptionArn=''
)

# Get list of subscriptions
response = sns.list_subscriptions_by_topic(
    TopicArn=''
)
subs = response['Subscriptions']

# Unsubscribe SMS subscrptions
for sub in subs:
    if sub['Protocol'] == 'sms':
        sns.unsubscribe(sub['SubscriptionsArn'])
