FROM apache/airflow:2.10.5

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow
RUN pip install --no-cache-dir \
    boto3 \
    python-dotenv \
    mistralai \
    selenium

# Pre-download docling models (optional, but recommended)
RUN python -c "from docling.document_converter import DocumentConverter; DocumentConverter()"
