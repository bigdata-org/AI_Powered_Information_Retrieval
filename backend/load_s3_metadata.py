from dotenv import load_dotenv
import os 
from ..airflow.dags.utils.aws.s3 import get_s3_client, metadataLinks
import json


load_dotenv()

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



if __name__ == "__main__":
    response = gets3url_metadata()
    print(response)