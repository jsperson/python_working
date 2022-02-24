import boto3
import jsp_params as params
firehose = boto3.client('firehose', region_name='us-east-1',
                        aws_access_key_id=params.AWS_KEY_ID,
                        aws_secret_access_key=params.AWS_SECRET)

record = {'record_id': '939ed1d1-1740-420c-8906-445278573c7f', 'timestamp': '14:25:06.000',
          'vin': '4FTEX4944AK844294', 'lon': 106.9447146, 'lat': -6.338565200000001, 'speed': 65}
payload = " ".join(str(value) for value in record.values())
# "939ed1d1-1740-420c-8906-445278573c7f 4:25:06.000 4FTEX4944AK844294 106.9447146 -6.

res = firehose.put_record(
    DeliveryStreamName='gps-delivery-stream',
    Record={'Data': payload + "\n"  # <-- Line break!
            }
)
