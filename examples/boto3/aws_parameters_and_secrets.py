import base64
import json
import boto3
from html import unescape
from botocore.config import Config
from botocore.exceptions import ClientError

# to use: from <this filename> import AWSServices as aws


class AWSServices():
    def config(region_name):
        return Config(
            region_name=region_name
        )

    def secretsmanager(region_name):
        config = AWSServices.config(region_name)
        return boto3.client(
            "secretsmanager",
            config=config
        )

    def get_secret(region_name, secret_name):
        client = AWSServices.secretsmanager(region_name)
        try:
            get_secret_value_response = client.get_secret_value(
                SecretId=secret_name
            )
        except ClientError as e:
            raise e
        else:
            # Decrypts secret using the associated KMS CMK.
            # Depending on whether the secret is a string or binary,
            # one of these fields will be populated.
            if "SecretString" in get_secret_value_response:
                secretString = get_secret_value_response["SecretString"]
                # replace entities, if present
                secretString = unescape(secretString)
                return json.loads(secretString)
            else:
                return json.loads(base64.b64decode(get_secret_value_response["SecretBinary"]))  # nopep8

    def systemsmanager(region_name):
        config = AWSServices.config(region_name)
        return boto3.client(
            "ssm",
            config=config
        )

    def get_string_parameter(region_name, parameter_name):
        ssm = AWSServices.systemsmanager(region_name)
        try:
            return ssm.get_parameter(Name=parameter_name)["Parameter"]["Value"]
        except ssm.exceptions.ParameterNotFound:
            print(f"Parameter not found for: {parameter_name}")
