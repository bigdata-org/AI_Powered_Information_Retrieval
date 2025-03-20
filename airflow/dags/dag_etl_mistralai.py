from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.databricks.operators.databricks import DatabricksRunNowOperator
from airflow.operators.empty import EmptyOperator
from utils.etl_mistralAi import metadataLinks, ocr_response, get_combined_markdown
from utils.check import check_metadata_present
from utils.scrape_links import scrape_pdf_links
from utils.aws.s3 import gets3url_metadata

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
    
    check_metadata = PythonOperator(
        task_id = "check_meatadata",
        python_callable= check_metadata_present
    )

    scrape_links =PythonOperator(
        task_id ="scrape_pdf_inks",
        python_callable= scrape_pdf_links
    )

    etl_mistralai = PythonOperator(
        task_id = "etl",
        python_callable= process_meatadata_links,
        trigger_rule = 'all_success'
    )

    generate_s3_url_metadata = PythonOperator(
        task_id = "s3_url_metadata",
        python_callable=gets3url_metadata,
        trigger_rule = "all_success"
    )

    run_notebook = DatabricksRunNowOperator(
        task_id = "run_notebook",
        databricks_conn_id="databricks_default",
        job_id="619557985935110",
        job_parameters={
            "tool" : "mistral"
        },
        trigger_rule = "all_success"
    )

    empty_task = EmptyOperator (
        task_id='empty_task'
    )


    scrape_links >> check_metadata
    check_metadata >> [etl_mistralai, empty_task]
    etl_mistralai >> generate_s3_url_metadata
    generate_s3_url_metadata >> run_notebook