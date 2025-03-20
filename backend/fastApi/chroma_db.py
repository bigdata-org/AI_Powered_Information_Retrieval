
import os
import boto3
import time
import numpy as np
import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from openai import OpenAI
import uvicorn
import logging

# Load environment variables
load_dotenv()

# Logging Configuration
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI()

AWS_ACCESS_KEY_ID = os.getenv("ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("REGION")
S3_BUCKET_NAME = os.getenv("BUCKET_NAME")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize AWS S3 Client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# Initialize OpenAI and Gemini
client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "nvidia_financial_reports"

try:
    collection = chroma_client.get_collection(name=collection_name)
    logging.info("ChromaDB collection loaded successfully.")
except Exception:
    collection = chroma_client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    logging.info("New ChromaDB collection created.")

def fetch_markdown_from_s3(year, quarter):
    file_key = f"{year}/{quarter}/mistral/nvidia_{quarter}.md"
    logging.debug(f"Fetching file from S3: {file_key}")

    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_key)
        content = response['Body'].read().decode('utf-8')
        if not content.strip():
            logging.warning(f"File {file_key} is empty.")
        return content
    except Exception as e:
        logging.error(f"Error fetching {file_key}: {str(e)}")
        return None

def chunk_text(text, strategy="default"):
    if not text:
        return []

    if strategy == "default":
        chunks = text.split("\n\n")
    elif strategy == "fixed":
        chunks = [text[i:i + 500] for i in range(0, len(text), 500)]
    elif strategy == "semantic":
        chunks = [text[i:i + 300] for i in range(0, len(text), 300)]
    else:
        chunks = [text]

    logging.debug(f"Chunking strategy: {strategy} | Total chunks created: {len(chunks)}")
    return [chunk for chunk in chunks if chunk.strip()]

def get_openai_embedding(text):
    if not text.strip():
        logging.warning("Attempted to get embedding for empty text.")
        return np.zeros(1536)

    try:
        response = client.embeddings.create(input=text, model="text-embedding-ada-002")
        return np.array(response.data[0].embedding, dtype=np.float32)
    except Exception as e:
        logging.error(f"OpenAI embedding error: {str(e)}")
        return np.zeros(1536)

def insert_all_data_to_chroma():
    logging.info("Starting data preprocessing and insertion into ChromaDB...")

    try:
        collection.delete(where={})
        logging.info("Cleared existing data from ChromaDB collection.")
    except Exception as e:
        logging.error(f"Error clearing collection: {str(e)}")

    for year in range(2021, 2025):
        for quarter in range(1, 5):
            markdown_content = fetch_markdown_from_s3(year, quarter)
            if markdown_content:
                for strategy in ["default", "fixed", "semantic"]:
                    content_chunks = chunk_text(markdown_content, strategy)
                    logging.debug(f"Processing Year {year}, Quarter {quarter}, Strategy {strategy}")

                    ids, embeddings, metadatas, documents = [], [], [], []
                    for i, chunk in enumerate(content_chunks):
                        if not chunk.strip():
                            continue
                        doc_id = f"{year}_{quarter}_{strategy}_{i}"
                        embedding = get_openai_embedding(chunk)

                        ids.append(doc_id)
                        embeddings.append(embedding.tolist())
                        metadatas.append({"year": str(year), "quarter": str(quarter), "strategy": strategy, "chunk_id": i})
                        documents.append(chunk)

                    if ids:
                        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
                        logging.info(f"Added {len(ids)} chunks with {strategy} strategy")

@app.post("/preprocess/")
def preprocess_data():
    try:
        insert_all_data_to_chroma()
        return {"message": "All data has been preprocessed and stored in ChromaDB."}
    except Exception as e:
        logging.error(f"Error during preprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during preprocessing: {str(e)}")

@app.get("/status/")
def get_status():
    try:
        count = collection.count()
        sample_results = collection.get(limit=10)

        years_available = set()
        quarters_available = set()
        strategies_available = set()

        for metadata in sample_results.get("metadatas", []):
            if metadata.get("year"):
                years_available.add(metadata["year"])
            if metadata.get("quarter"):
                quarters_available.add(metadata["quarter"])
            if metadata.get("strategy"):
                strategies_available.add(metadata["strategy"])

        return {
            "status": "active",
            "document_count": count,
            "years_available": list(years_available),
            "quarters_available": list(quarters_available),
            "strategies_available": list(strategies_available)
        }
    except Exception as e:
        logging.error(f"Error getting status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")

if __name__ == "__main__":
    logging.info("Starting FastAPI server...")
    uvicorn.run("chroma_db:app", host="0.0.0.0", port=8001, reload=True)
