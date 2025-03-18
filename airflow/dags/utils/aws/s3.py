import boto3
import os
from dotenv import load_dotenv


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