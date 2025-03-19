from datetime import datetime
from dotenv import load_dotenv
from airflow import DAG
from dotenv import load_dotenv
from airflow.operators.python import PythonOperator
from utils.scrape_links import scrape_pdf_links



load_dotenv()


with DAG(
    dag_id='dag_scrape_links',
    description='scraping lonk to pdf files',
    start_date=datetime(2025,3,7),
    schedule_interval='@monthly'
) as dag :
    
    scrape_pdf_links = PythonOperator(
        task_id = 'Selenium_scrapper',
        python_callable=scrape_pdf_links
    )

    scrape_pdf_links