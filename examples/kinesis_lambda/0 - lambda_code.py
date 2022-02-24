import json
import boto3
import pandas as pd
import os

SPEED_ALERT_THRESHOLD = int(os.environ.get('SPEED_ALERT_THRESHOLD', 45))
ALERT_PHONE_NUMBER = os.environ.get('ALERT_PHONE_NUMBER', None)


def record_created_handler(event, context):
    sns = boto3.client('sns')

    # Call the helper method
    data = get_new_data(event)
    # Get the top speeds
    top_speeds = data.groupby(['vin'])['speed'].max().reset_index()
    # Get top speeds that exceed the limit of 45
    too_fast = top_speeds.loc[top_speeds.speed > SPEED_ALERT_THRESHOLD, :]

    print(too_fast)

    # Send SMS
    sns.publish(PhoneNumber=ALERT_PHONE_NUMBER,
                Message="Speeding Alert \n" + too_fast.to_string())

    # This doesn't go anywhere yet, but we need to return something.
    totals = data.groupby(['vin'])['speed'].max().reset_index()
    return totals.to_csv(sep=" ", index=False)


def get_new_data(event):
    # Create a list to store new object keys
    written_objects = []

    # Iterate over each S3 event record.
    for record in event['Records']:
        s3 = boto3.client('s3')
        # Get the variables to check for
        event_name = record['eventName']
        bucket_name = record['s3']['bucket']['name']
        obj_key = record['s3']['object']['key']

        # Verify that event is created from sd-vehicle-data bucket.
        if event_name == 'ObjectCreated:Put' and bucket_name == 'sd-vehicle-data-jsp':
            obj = s3.get_object(Bucket=bucket_name, Key=obj_key)
            df = pd.read_csv(obj['Body'], delimiter=" ",
                             names=["record_id", "timestamp", "vin", "lon", "lat", "speed"])
            # Concatenate new records into a single dataframe.
            written_objects.append(df)

    return pd.concat(written_objects)
