from mistralai import Mistral
import os 
from mistralai.models import OCRResponse
from dotenv import load_dotenv
from fastApi.utils.aws.s3 import get_s3_client
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

def replace_images_in_markdown(markdown_str: str, images_dict: dict, filename: str) -> str:

    for img_name, base64_str in images_dict.items():
        img_data = base64_str.split(';')[1].split(',')[1]
        image_data = base64.b64decode(img_data)
        s3_image_key = f"uploads/mistral/{filename}/images/{img_name}"  

        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_image_key,
            Body=image_data,
            ContentType="image/png"
        )
        s3_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_image_key}"
        markdown_str = markdown_str.replace(f"![{img_name}]({img_name})", f"![{img_name}]({s3_url})")
 
    return markdown_str

def get_combined_markdown(ocr_response: OCRResponse, filename: str) -> str:
    markdowns: list[str] = []
    # Extract images from page
    try:
        for page in ocr_response.pages:
            image_data = {}
            for img in page.images:
                image_data[img.id] = img.image_base64
            # Replace image placeholders with actual images
            markdowns.append(replace_images_in_markdown(page.markdown, image_data, filename))
        md_file = "\n\n".join(markdowns)

        s3_key = f"uploads/mistral/{filename}/{filename}.md"
            
        s3_client.put_object(
        Bucket=bucket_name,
        Key=s3_key,
        Body=md_file.encode("utf-8"),  
        ContentType="text/markdown" 
        )
            
        return f"file uploaded to s3: {filename}"
    except Exception as e:
        return str(e)


def mistral_parser(url):
    file_name = url.split("/")[-1].split(".")[0]
    ocrResponse = ocr_response(url)
    response = get_combined_markdown(ocrResponse,file_name)
    return response
    


# if __name__ == "__main__":
#     respones = mistral_parser("https://rag-pipeline-data.s3.us-east-2.amazonaws.com/p.pdf")
#     print(respones)
            



            
