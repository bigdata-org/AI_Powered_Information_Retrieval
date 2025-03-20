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


def gets3url_metadata():
    s3_client = get_s3_client()
    bucket_name = os.getenv('BUCKET_NAME')
    region = os.getenv('REGION')
    data = metadataLinks()
    metadata_s3 = {}
    try:
        for year, qtrs in data.items():
            if year not in metadata_s3:
                metadata_s3[year] = {}  # Initialize year key
            for qtr, link in qtrs.items():
                metadata_s3[year][qtr] = {
                    "docling": f"https://{bucket_name}.s3.{region}.amazonaws.com/{year}/{qtr}/docling/nvidia_report{year}.md",
                    "mistralAi": f"https://{bucket_name}.s3.{region}.amazonaws.com/{year}/{qtr}/mistral/nvidia_{qtr}.md"
                }
        

        metadata_json = json.dumps(metadata_s3, indent=4)
        json_s3_key = "metadata/metadata_s3url.json"

        s3_client.put_object(
            Bucket=bucket_name,
            Key=json_s3_key,
            Body=metadata_json.encode("utf-8"),
            ContentType="application/json"
        )
        return "s3 url metadata uploaded"
    except Exception as e:
        return str(e)

# if __name__ == "__main__":
#     gets3url_metadata()
