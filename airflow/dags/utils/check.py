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



def check_s3url_json_is_updated():
    s3_client = s3.get_s3_client()
    response = s3_client.list_objects_v2(Bucket=bucket_name, Delimiter='/')
    folders = [prefix["Prefix"].split("/")[0] for prefix in response.get("CommonPrefixes", [])]
    years_only = list(filter(lambda x: x.isdigit() and len(x) == 4, folders))

    # check years in meatadata_s3url.json
    data = s3.metadataLinks()
    years = []
    for key, val in data.items():
        years.append(key)

    return years_only == years 
    
