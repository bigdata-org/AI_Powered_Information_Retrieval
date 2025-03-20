

import os
import boto3
import time
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from openai import OpenAI
import uvicorn

# Load environment variables
load_dotenv()

app = FastAPI()

AWS_ACCESS_KEY_ID = os.getenv("ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("REGION")
S3_BUCKET_NAME = os.getenv("BUCKET_NAME")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
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

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "nvidia-financial-reports"

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
    print(f"Created new Pinecone index: {index_name}")
    time.sleep(30)  # Wait for index to be created

index = pc.Index(index_name)

def generate_gemini_response(context, query):
    """Generate a response using Gemini API."""
    prompt = f"""You are a financial analyst assistant. Use this context to answer the question:
    
    Context:
    {context}
    
    Question: {query}
    
    Answer in clear, concise bullet points with factual information only:"""
    
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating Gemini response: {str(e)}")
        return "I'm sorry, but I couldn't generate a response at the moment."

def fetch_markdown_from_s3(year, quarter):
    """Fetch markdown file from S3 for the given year & quarter."""
    file_key = f"{year}/{quarter}/mistral/nvidia_{quarter}.md"
    
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_key)
        content = response['Body'].read().decode('utf-8')
        if not content.strip():
            print(f"Warning: File {file_key} is empty.")
        return content
    except s3_client.exceptions.NoSuchKey:
        print(f"File not found: {file_key}")
        return None
    except s3_client.exceptions.ClientError as e:
        print(f"AWS S3 ClientError for {file_key}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error fetching {file_key}: {e}")
        return None

def chunk_text(text, strategy="default"):
    """Chunk text based on the selected strategy."""
    if not text:
        return []
        
    if strategy == "default":
        chunks = text.split("\n\n")  # Simple paragraph split
    elif strategy == "fixed":
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    elif strategy == "semantic":
        chunks = [text[i:i+300] for i in range(0, len(text), 300)]
    else:
        chunks = [text]
        
    # Filter out empty chunks
    return [chunk for chunk in chunks if chunk.strip()]

def check_if_data_exists():
    """Check if data already exists in Pinecone index."""
    try:
        # Query for a sample document to check if data exists
        sample_result = index.query(
            vector=[0.0] * 1536,  # Dummy vector
            filter={"strategy": {"$eq": "default"}},
            top_k=1
        )
        return len(sample_result.get('matches', [])) > 0
    except Exception as e:
        print(f"Error checking if data exists: {str(e)}")
        return False

def insert_all_data_to_pinecone():
    """Efficiently preprocess all data from S3 and store in Pinecone using batching."""
    print("Starting data preprocessing...")
    
    # Define years and quarters to process
    years = range(2021, 2025)
    quarters = range(1, 4)
    strategies = ["default", "fixed", "semantic"]
    
    total_combinations = len(years) * len(quarters) * len(strategies)
    processed = 0
    
    for strategy in strategies:
        print(f"Processing chunking strategy: {strategy}")
        
        for year in years:
            for quarter in quarters:
                processed += 1
                print(f"Processing {year} Q{quarter} with {strategy} strategy ({processed}/{total_combinations})")
                
                # Fetch markdown content
                markdown_content = fetch_markdown_from_s3(year, quarter)
                if not markdown_content:
                    print(f"No content found for {year} Q{quarter}, skipping...")
                    continue
                
                # Chunk the content
                content_chunks = chunk_text(markdown_content, strategy)
                if not content_chunks:
                    print(f"No chunks generated for {year} Q{quarter}, skipping...")
                    continue
                
                print(f"Generated {len(content_chunks)} chunks for {year} Q{quarter}")
                
                # Process in batches
                batch_size = 50  # Adjust based on API limits
                for i in range(0, len(content_chunks), batch_size):
                    batch = content_chunks[i:i+batch_size]
                    
                    try:
                        print(f"Processing batch {i//batch_size + 1}/{(len(content_chunks) + batch_size - 1)//batch_size}")
                        
                        # Get embeddings in batch
                        response = client.embeddings.create(
                            input=batch,
                            model="text-embedding-ada-002"
                        )
                        
                        # Prepare vectors for batch upsert
                        vectors = []
                        for j, embedding_data in enumerate(response.data):
                            chunk_idx = i + j
                            doc_id = f"{year}_{quarter}_{strategy}_{chunk_idx}"
                            
                            vectors.append((
                                doc_id, 
                                embedding_data.embedding, 
                                {
                                    "year": str(year),
                                    "quarter": str(quarter),
                                    "strategy": strategy,
                                    "content": batch[j]
                                }
                            ))
                        
                        # Batch upsert to Pinecone
                        if vectors:
                            index.upsert(vectors=vectors)
                            print(f"Successfully upserted {len(vectors)} vectors")
                        
                    except Exception as e:
                        print(f"Error processing batch starting at chunk {i}: {str(e)}")
    
    print("Data preprocessing complete!")

# Check if data exists before preprocessing
if not check_if_data_exists():
    print("No existing data found in Pinecone. Starting preprocessing...")
    insert_all_data_to_pinecone()
else:
    print("Data already exists in Pinecone. Skipping preprocessing.")

def query_pinecone(query, years=None, quarters=None, strategy="default", top_k=5):
    """Query Pinecone for relevant data."""
    try:
        query_embedding = client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        ).data[0].embedding
    except Exception as e:
        print(f"Embedding generation failed: {str(e)}")
        return []

    # Build filter conditions
    filter_conditions = {"strategy": {"$eq": strategy}}
    
    if years and len(years) > 0:
        year_strs = [str(year) for year in years]
        filter_conditions["year"] = {"$in": year_strs}
        
    if quarters and len(quarters) > 0:
        quarter_strs = [str(quarter) for quarter in quarters]
        filter_conditions["quarter"] = {"$in": quarter_strs}

    try:
        result = index.query(
            vector=query_embedding,
            filter=filter_conditions,
            top_k=top_k,
            include_metadata=True
        )
        return result.get('matches', [])
    except Exception as e:
        print(f"Pinecone query failed: {str(e)}")
        return []

class QueryRequest(BaseModel):
    query: str
    years: list[int]  # Allow multiple years
    quarters: list[int]  # Allow multiple quarters
    chunking_strategy: str = "default"  # User selects chunking strategy
    top_k: int = 5  # Number of results to return

@app.post("/query/")
def query_data(request: QueryRequest):
    """Query the system for financial report insights."""
    # Validate input
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    if not request.years or not request.quarters:
        raise HTTPException(status_code=400, detail="At least one year and quarter must be specified")
    
    # Query Pinecone
    pinecone_results = query_pinecone(
        request.query, 
        request.years, 
        request.quarters, 
        request.chunking_strategy,
        request.top_k
    )

    if not pinecone_results:
        raise HTTPException(status_code=404, detail="No relevant data found in Pinecone.")

    # Extract the content from the results
    context = "\n\n".join([match['metadata']['content'] for match in pinecone_results])
    
    # Generate response using Gemini
    gemini_response = generate_gemini_response(context, request.query)

    return {
        "query": request.query,
        "response": gemini_response,
        "matches": len(pinecone_results),
        "years": request.years,
        "quarters": request.quarters,
        "chunking_strategy": request.chunking_strategy
    }

@app.get("/health")
def health_check():
    """API health check endpoint."""
    return {"status": "healthy", "pinecone_index": index_name}

if __name__ == "__main__":
    uvicorn.run("pinecone_db:app", host="0.0.0.0", port=8000, reload=True)





