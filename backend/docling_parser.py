from fastApi.utils.aws import s3
import os 
from dotenv import load_dotenv
from boto3.s3.transfer import TransferConfig
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption,InputFormat
import re
from io import BytesIO


load_dotenv()

def upload_data_etl(url):
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 1.0

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    headers = {
        'User-Agent': 'MIT  bigdata@gmail.com',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Host': 'www.sec.gov',
        'Connection': 'keep-alive'
        }

    bucket_name = os.getenv('BUCKET_NAME')
    region = os.getenv('REGION')
    s3_client = s3.get_s3_client()

    file_name = url.split("/")[-1].split(".")[0]

    # convert to md
    source = url # document per local path or URL
    result = converter.convert(source)
    md_content = result.document.export_to_markdown()

    for i, picture in enumerate(result.document.pictures):
        image_data = picture.get_image(result.document)
        if image_data:  # Ensure image exists
            md_content = re.sub("<!-- image -->", f"<!-- image_{i+1} -->", md_content, count=1)
        
            # local_image_path = os.path.join(output_img, f"image_{i + 1}.png")
            s3_key = f"uploads/docling/{file_name}/images/image_{i + 1}.png"

            # image_data.save(local_image_path)
            img_buffer = BytesIO()
            image_data.save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()

            try:
                s3_client.put_object(
                    Body=img_bytes,
                    Bucket=bucket_name, 
                    Key=s3_key, 
                    ContentType='image/png'
                    )
                print("images uploaded")
            except Exception as e:
                print(f"Images not uploaded:{str(e)}")

            s3_img_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_key}"
            md_content = md_content.replace(f"<!-- image_{i+1} -->", f"![Image {i + 1}]({s3_img_url})")  
        else:
            print("no img found")

    md_bytes = BytesIO(md_content.encode("utf-8"))
    s3_key_md = f"uploads/docling/{file_name}/{file_name}.md"
    try:
        s3_client.put_object(
                Body=md_bytes.getvalue(),
                Bucket=bucket_name, 
                Key=s3_key_md, 
                ContentType='text/markdown'
                )
        print(f"{file_name} file uploaded ")
    except :
        print("file not uploaded")

    return f"All files uploaded to s3"

# if __name__ == "__main__":
#     upload_data_etl("https://rag-pipeline-data.s3.us-east-2.amazonaws.com/p.pdf")