from datetime import datetime
from dotenv import load_dotenv
from airflow import DAG
from dotenv import load_dotenv
from airflow.operators.python import PythonOperator
from utils.etl_docling import report_data_etl





with DAG(
    dag_id="dag_etl_docling",
    description="dag to extract pdf convert it to md using docling and load int s3",
    schedule_interval="@monthly",
    
) as dag :
    
    etl_docling = PythonOperator(
        task_id = "etl_docling",
        python_callable= report_data_etl
    )

    etl_docling