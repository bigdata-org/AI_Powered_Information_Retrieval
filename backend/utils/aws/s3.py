import boto3
import os
from dotenv import load_dotenv
from io import BytesIO, StringIO
import json
import requests

load_dotenv() 

def get_s3_client():
    try:
        s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION")  
        )
        return s3_client
    except:
        return -1

def upload_pdf_to_s3(s3_client, file_name, file_bytes_io: BytesIO):
    try:
        bucket_name, aws_region = os.getenv("BUCKET_NAME"), os.getenv('AWS_REGION')
        if bucket_name is None or aws_region is None:
            return -1
        # Define S3 file path
        s3_file_path = f"uploads/{file_name}.pdf"
        # Upload the file to S3 using upload_fileobj
        s3_client.upload_fileobj(file_bytes_io, bucket_name, s3_file_path)

        # Construct the public URL for the uploaded file
        object_url = f"https://{bucket_name}.s3.{aws_region}.amazonaws.com/{s3_file_path}"
        return object_url
    except Exception as e:
        return -1

