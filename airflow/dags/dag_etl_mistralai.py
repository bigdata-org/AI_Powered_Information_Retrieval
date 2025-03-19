from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from utils.etl_mistralAi import metadataLinks, ocr_response, get_combined_markdown


def process_meatadata_links():
    data = metadataLinks()
    result = []
    for year, qtrs in data.items():
        for qtr, link in data[year].items():
            ocrResponse = ocr_response(link)
            response = get_combined_markdown(ocrResponse,year,qtr)
            result.append(response)

    return result            
    


with DAG(
    dag_id="dag_etl_mistralAi",
    description="dag to extract pdf convert it to md using mistralAi and load int s3",
    start_date=datetime(2025,3,7),
    schedule_interval="@monthly"
    
) as dag :
    
    etl_mistralAi = PythonOperator(
        task_id = "etl_mistralAi",
        python_callable= process_meatadata_links
    )

    etl_mistralAi