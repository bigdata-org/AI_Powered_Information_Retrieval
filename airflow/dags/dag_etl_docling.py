from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.databricks.operators.databricks import DatabricksRunNowOperator
from airflow.operators.empty import EmptyOperator
from utils.check import check_metadata_present
from utils.scrape_links import scrape_pdf_links
from utils.aws.s3 import gets3url_metadata
from airflow.operators.bash import BashOperator
import json
import os
from dotenv import load_dotenv
from utils.check import check_s3url_json_is_updated


load_dotenv()

with DAG(
    dag_id="dag_etl_docling",
    description="dag to extract pdf convert it to md using docling and load int s3",
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

    # run_notebook = DatabricksRunNowOperator(
    #     task_id = "run_notebook",
    #     databricks_conn_id="databricks_default",
    #     job_id="619557985935110",
    #     notebook_params={
    #         "tool" : "docling"
    #     }
    # )

    trigger_github_workflow = BashOperator(
        task_id='trigger_github_workflow',
        bash_command= """
            curl -X POST \
            -H "Accept: application/vnd.github+json" \
            -H "Authorization: Bearer $GITHUB_BEARER_TOKEN" \
            https://api.github.com/repos/bigdata-org/AI_Powered_Information_Retrieval/actions/workflows/etl_docling.yaml/dispatches \
            -d '{"ref": "main"}'
        """,
        env={"GITHUB_BEARER_TOKEN": "{{ var.value.github_bearer_token }}"}
    )

    empty_task = EmptyOperator (
        task_id='empty_task'
    )

    scrape_links >> check_metadata
    check_metadata >> [trigger_github_workflow, empty_task]
 