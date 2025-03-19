import boto3
import os
from dotenv import load_dotenv
import requests
import json

load_dotenv() 
def get_s3_client():
    try:
        s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv("ACCESS_KEY"),
        aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
        region_name=os.getenv("REGION")  
        )
        return s3_client
    except:
        return -1
    


def metadataLinks():
    bucket_name = os.getenv('BUCKET_NAME')
    region = os.getenv('REGION')

    json_link_key = "metadata/metadata.json"
    s3_url =f"https://{bucket_name}.s3.{region}.amazonaws.com/{json_link_key}"

    response = requests.get(s3_url)
    json_data = json.loads(response.content)

    return json_data