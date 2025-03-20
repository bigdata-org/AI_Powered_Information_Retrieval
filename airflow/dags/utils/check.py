import os 
from dotenv import load_dotenv
from utils.aws import s3

load_dotenv()

bucket_name = os.getenv('BUCKET_NAME')

def check_metadata_present():
    s3_client = s3.get_s3_client()
    s3_key = "metadata/metadata.json"
    try :
        s3_client.head_object(Bucket=bucket_name,Key = s3_key)
        return "etl"
    except:
        return "empty_task"


