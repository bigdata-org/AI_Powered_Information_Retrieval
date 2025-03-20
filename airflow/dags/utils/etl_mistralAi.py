from mistralai import Mistral
import os 
from mistralai.models import OCRResponse
from dotenv import load_dotenv
from utils.aws.s3 import get_s3_client, metadataLinks
import base64
from mistralai.models import SDKError
import time 
load_dotenv()


api_key = os.getenv('MISTRAL_API_KEY')
client = Mistral(api_key=api_key)
s3_client = get_s3_client()
bucket_name = os.getenv('BUCKET_NAME')
region = os.getenv('REGION')
# text_model = "mistral-small-latest"
# ocr_model = "mistral-ocr-latest"



def ocr_response(url, retries=3, delay=5):
   for attempt in range(retries):
        try:
            return client.ocr.process(
                model="mistral-ocr-latest",
                document={"type": "document_url", "document_url": url},
                include_image_base64=True
            )
        except SDKError as e:
            if "429" in str(e):
                print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                time.sleep(delay)  # Wait before retrying
                delay *= 2  # Exponential backoff
            else:
                raise e  # Raise error if it's not rate limit-related

   raise SDKError("Max retries reached. Please try again later.")

def replace_images_in_markdown(markdown_str: str, images_dict: dict, year: int, qtr: str) -> str:

    for img_name, base64_str in images_dict.items():
        img_data = base64_str.split(';')[1].split(',')[1]
        image_data = base64.b64decode(img_data)
        s3_image_key = f"{year}/{qtr}/mistral/images/{img_name}"  

        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_image_key,
            Body=image_data,
            ContentType="image/png"
        )
        s3_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_image_key}"
        markdown_str = markdown_str.replace(f"![{img_name}]({img_name})", f"![{img_name}]({s3_url})")
 
    return markdown_str

def get_combined_markdown(ocr_response: OCRResponse, year: int , qtr: str) -> str:
    markdowns: list[str] = []
    # Extract images from page
    try:
        for page in ocr_response.pages:
            image_data = {}
            for img in page.images:
                image_data[img.id] = img.image_base64
            # Replace image placeholders with actual images
            markdowns.append(replace_images_in_markdown(page.markdown, image_data, year, qtr))
        md_file = "\n\n".join(markdowns)

        s3_key = f"{year}/{qtr}/mistral/nvidia_{qtr}.md"
            
        s3_client.put_object(
        Bucket=bucket_name,
        Key=s3_key,
        Body=md_file.encode("utf-8"),  
        ContentType="text/markdown" 
        )
            
        return f"file uploaded to s3: {qtr}"
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    data = metadataLinks()
    for year, qtrs in data.items():
        for qtr, link in data[year].items():
            ocrResponse = ocr_response(link)
            response = get_combined_markdown(ocrResponse,year,qtr)
            
            print(response)

            
            




            
