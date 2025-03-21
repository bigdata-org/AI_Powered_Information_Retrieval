from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.databricks.operators.databricks import DatabricksRunNowOperator
from airflow.operators.empty import EmptyOperator
from utils.etl_mistralAi import metadataLinks, ocr_response, get_combined_markdown
from utils.check import check_metadata_present
from utils.scrape_links import scrape_pdf_links
from utils.aws.s3 import gets3url_metadata


with DAG(
    dag_id='dag_test',
    description='scraping lonk to pdf files',
    start_date=datetime(2025,3,7),
    schedule_interval='@monthly'
) as dag :
    
    run_notebook = DatabricksRunNowOperator(
        task_id = "run_notebook",
        databricks_conn_id="databricks_default",
        job_id="915262908716993",
        job_parameters={
            "tool" : "mistral"
        },
        trigger_rule = "all_success"
    )



# # Add a task to check the status
# check_workflow_status = BashOperator(
#     task_id='check_workflow_status',
#     bash_command= """
#         # Extract run ID from the URL
#         RUN_URL=$(cat /tmp/workflow_run_url.txt)
        
#         # Poll until complete
#         while true; do
#             STATUS=$(curl -s -H "Authorization: Bearer $GITHUB_BEARER_TOKEN" $RUN_URL | jq -r '.conclusion')
#             if [[ "$STATUS" != "null" ]]; then
#                 echo "Workflow completed with status: $STATUS"
#                 break
#             fi
#             sleep 30
#         done
#     """,
#     env={"GITHUB_BEARER_TOKEN": "{{ var.value.github_bearer_token }}"}
# )