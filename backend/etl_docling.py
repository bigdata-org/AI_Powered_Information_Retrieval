from fastApi.utils.aws import s3
import os 
from dotenv import load_dotenv
import requests
import json
from boto3.s3.transfer import TransferConfig
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption,InputFormat
import re
from io import BytesIO


load_dotenv()

def report_data_etl():
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 1.0

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # def store_data_to_s3():
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

    json_link_key = "metadata/metadata.json"
    s3_url =f"https://{bucket_name}.s3.{region}.amazonaws.com/{json_link_key}"

    print(f"loading data from {s3_url}")
    response = requests.get(s3_url)
    json_data = json.loads(response.content)


    try:
        for year,qtrs in json_data.items():
            for qtr, link in json_data[year].items():
                file_link = json_data[year][qtr]

                print(f"loadind data for year {year}: {qtr}")
                # convert to md
                source = file_link # document per local path or URL
                result = converter.convert(source)
                md_content = result.document.export_to_markdown()


                md_file = f"nvidia_report{year}"
                for i, picture in enumerate(result.document.pictures):
                    image_data = picture.get_image(result.document)
                    if image_data:  # Ensure image exists
                        md_content = re.sub("<!-- image -->", f"<!-- image_{i+1} -->", md_content, count=1)
                    
                        # local_image_path = os.path.join(output_img, f"image_{i + 1}.png")
                        s3_key = f"{year}/{qtr}/docling/images/image_{i + 1}.png"

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
                        except Exception as e:
                            print(f"Images not uploaded:str(e)")

                        s3_img_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_key}"
                        md_content = md_content.replace(f"<!-- image_{i+1} -->", f"![Image {i + 1}]({s3_img_url})")  
                    else:
                        print("no img found")

                md_bytes = BytesIO(md_content.encode("utf-8"))
                s3_key_md = f"{year}/{qtr}/docling/{md_file}.md"
                try:
                    s3_client.put_object(
                            Body=md_bytes.getvalue(),
                            Bucket=bucket_name, 
                            Key=s3_key_md, 
                            ContentType='text/markdown'
                            )
                    print(f"{year} {qtr}: {file_link}file uploaded ")
                except :
                    print("file not uploaded")
        
        return f"All files uploaded to s3 {year}"
        
    except  Exception as e:
        return {f"Error": str(e)}




if __name__ == "__main__":
    response = report_data_etl()
    print(response)