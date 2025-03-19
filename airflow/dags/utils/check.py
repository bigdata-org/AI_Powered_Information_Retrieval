from firecrawl import FirecrawlApp
import os 
from dotenv import load_dotenv
import json
from utils.aws import s3

load_dotenv()

bucket_name = os.getenv('S3_BUCKET_NAME')
def check_if_file_exists(**context):
    year = context['params'].get('year','2024')
    qtr = context['params'].get('qtr','4')

    ti = context['task_instance']
    ti.xcom_push(key='year', value=year)
    ti.xcom_push(key='qtr', value=qtr)

    s3_client = s3.get_s3_client()
    response = s3_client.list_objects_v2(
        Bucket=bucket_name,
        Prefix=f'sec_data/{year}/{qtr}/'
    )
    Keys = [obj['Key'] for obj in response.get('Contents', [])]
    file_count = len(Keys)

    if file_count >= 4:
        return 'dbt_curl_command'
    return 'Upload_data_to_s3'
