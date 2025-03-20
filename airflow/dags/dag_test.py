from datetime import datetime
from dotenv import load_dotenv
from airflow import DAG
from airflow.operators.bash import BashOperator



load_dotenv()

with DAG(
    dag_id='dag_test',
    description='scraping lonk to pdf files',
    start_date=datetime(2025,3,7),
    schedule_interval='@monthly'
) as dag :
    
    check_workflow_status = BashOperator(
    task_id='check_workflow_status',
    bash_command= """
        # Extract run ID from the URL
        RUN_URL=$(cat /tmp/workflow_run_url.txt)
        
        # Poll until complete
        while true; do
            STATUS=$(curl -s -H "Authorization: Bearer $GITHUB_BEARER_TOKEN" $RUN_URL | jq -r '.conclusion')
            if [[ "$STATUS" != "null" ]]; then
                echo "Workflow completed with status: $STATUS"
                break
            fi
            sleep 30
        done
    """,
    env={"GITHUB_BEARER_TOKEN": "{{ var.value.github_bearer_token }}"}
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