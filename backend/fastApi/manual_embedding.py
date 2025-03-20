# import os
# import torch
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# # Load the Sentence Transformer model
# model = SentenceTransformer("all-MiniLM-L6-v2")  

# # Load and process the markdown file
# def load_markdown_file(file_path):
#     with open(file_path, "r", encoding="utf-8") as file:
#         content = file.read()
#     sentences = content.split("\n") 
#     return [s.strip() for s in sentences if s.strip()] 

# # Compute embeddings for the document
# def compute_embeddings(sentences):
#     return model.encode(sentences, convert_to_tensor=True)

# # Find the most similar text using cosine similarity
# def find_similar_text(query, sentences, sentence_embeddings, top_k=3):
#     query_embedding = model.encode(query, convert_to_tensor=True)
#     similarities = cosine_similarity(query_embedding.cpu().numpy().reshape(1, -1), sentence_embeddings.cpu().numpy())
    
#     top_k_indices = np.argsort(similarities[0])[::-1][:top_k]  # Get top-k highest similarity indices
#     return [sentences[i] for i in top_k_indices]


# file_path = "nvidia_q1_2025.md"

# # Load, encode, and store embeddings
# sentences = load_markdown_file(file_path)
# sentence_embeddings = compute_embeddings(sentences)

# # Example Query
# query = "What is cashflow summary?"
# top_matches = find_similar_text(query, sentences, sentence_embeddings)

# # Output results
# print("\nTop Relevant Sentences:")
# for match in top_matches:
#     print(f"- {match}")








# import os
# import boto3
# import redis
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel

# # Initialize Redis client
# redis_client = redis.Redis(host='localhost', port=6379, db=0)

# # Load the Sentence Transformer model
# manual_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Function to load markdown file from S3
# def load_markdown_file_from_s3(s3_client, bucket_name, file_key):
#     try:
#         response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
#         content = response['Body'].read().decode('utf-8')
#         if not content.strip():
#             print(f" Warning: File {file_key} is empty.")
#         return content
#     except Exception as e:
#         print(f"Error loading file {file_key}: {e}")
#         return None

# # Compute embeddings for a document
# def compute_manual_embeddings(text):
#     sentences = text.split("\n\n")  # Split into paragraphs
#     sentences = [s.strip() for s in sentences if s.strip()]
#     return manual_model.encode(sentences, convert_to_tensor=True)

# # Chunking strategies
# def chunk_text(text, strategy="default"):
#     if strategy == "default":
#         return text.split("\n\n")  # Simple paragraph split
#     elif strategy == "fixed":
#         return [text[i:i+500] for i in range(0, len(text), 500)]
#     elif strategy == "semantic":
#         return [text[i:i+300] for i in range(0, len(text), 300)]  
#     return [text]

# # Preprocess data for manual embedding
# def preprocess_manual_data(s3_client, bucket_name, year, quarter):
#     file_key = f"{year}/{quarter}/mistral/nvidia_{quarter}.md"
#     markdown_content = load_markdown_file_from_s3(s3_client, bucket_name, file_key)
#     if markdown_content:
#         for strategy in ["default", "fixed", "semantic"]:
#             content_chunks = chunk_text(markdown_content, strategy)
#             yield strategy, content_chunks

# # Store preprocessed data in Redis
# def store_manual_data_in_redis(year, quarter, strategy, chunks):
#     key = f"{year}_{quarter}_{strategy}"
#     redis_client.set(key, str(chunks))

# # Fetch preprocessed data from Redis
# def fetch_manual_data_from_redis(year, quarter, strategy):
#     key = f"{year}_{quarter}_{strategy}"
#     return eval(redis_client.get(key))

# # Preprocess all data and store in Redis
# def preprocess_and_store_all_data(s3_client, bucket_name):
#     for year in range(2021, 2026): 
#         for quarter in range(1, 5):
#             for strategy, chunks in preprocess_manual_data(s3_client, bucket_name, year, quarter):
#                 store_manual_data_in_redis(year, quarter, strategy, chunks)

# # Initialize AWS S3 Client
# AWS_ACCESS_KEY_ID = os.getenv("ACCESS_KEY")
# AWS_SECRET_ACCESS_KEY = os.getenv("SECRET_ACCESS_KEY")
# AWS_REGION = os.getenv("REGION")
# S3_BUCKET_NAME = os.getenv("BUCKET_NAME")

# s3_client = boto3.client(
#     "s3",
#     aws_access_key_id=AWS_ACCESS_KEY_ID,
#     aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#     region_name=AWS_REGION
# )

# # Preprocess and store all data
# preprocess_and_store_all_data(s3_client, S3_BUCKET_NAME)

# # Find the most similar text using cosine similarity
# def find_similar_text_manual(query, sentences, sentence_embeddings, top_k=3):
#     query_embedding = manual_model.encode(query, convert_to_tensor=True)
#     similarities = cosine_similarity(query_embedding.cpu().numpy().reshape(1, -1), sentence_embeddings.cpu().numpy())
    
#     top_k_indices = np.argsort(similarities[0])[::-1][:top_k]  # Get top-k highest similarity indices
#     return [sentences[i] for i in top_k_indices]

# # Query manual data
# def query_manual_data(query, year, quarter, strategy="default", top_k=3):
#     chunks = fetch_manual_data_from_redis(year, quarter, strategy)
#     if chunks:
#         sentence_embeddings = manual_model.encode(chunks, convert_to_tensor=True)
#         return find_similar_text_manual(query, chunks, sentence_embeddings, top_k)
#     else:
#         return []

# # FastAPI App
# app = FastAPI()

# class QueryRequest(BaseModel):
#     query: str
#     years: list[int]  # Allow multiple years
#     quarters: list[int]  # Allow multiple quarters
#     chunking_strategy: str = "default"  # User selects chunking strategy
#     embedding_type: str = "pinecone"  # Options: pinecone, chroma, manual

# @app.post("/query/")
# def query_data(request: QueryRequest):
#     results = []
#     for year in request.years:
#         for quarter in request.quarters:
#             if request.embedding_type == "manual":
#                 manual_matches = query_manual_data(request.query, year, quarter, request.chunking_strategy)
#                 results.extend([{"content": match} for match in manual_matches])
#             # Implement Pinecone and Chroma DB logic here
    
#     if not results:
#         raise HTTPException(status_code=404, detail="No relevant data found.")

#     context = "\n\n".join([match.get('metadata', {}).get('content', match.get('content', '')) for match in results])
#     # Assuming gemini_response generation logic is elsewhere
#     gemini_response = "Example Response"  # Replace with actual logic

#     return {"query": request.query, "response": gemini_response, "matches": len(results)}







# import os
# import boto3
# import redis
# import pickle
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import List, Dict, Any, Optional

# # Initialize FastAPI app
# app = FastAPI(title="NVIDIA Financial Records Semantic Search")

# # Initialize Redis client with connection pooling for better performance
# redis_pool = redis.ConnectionPool(host='localhost', port=6379, db=0)
# redis_client = redis.Redis(connection_pool=redis_pool)

# # Load the Sentence Transformer model once for reuse
# manual_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Initialize AWS S3 Client
# def get_s3_client():
#     AWS_ACCESS_KEY_ID = os.getenv("ACCESS_KEY")
#     AWS_SECRET_ACCESS_KEY = os.getenv("SECRET_ACCESS_KEY")
#     AWS_REGION = os.getenv("REGION")
    
#     if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION]):
#         raise ValueError("AWS credentials not properly set in environment variables")
    
#     return boto3.client(
#         "s3",
#         aws_access_key_id=AWS_ACCESS_KEY_ID,
#         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#         region_name=AWS_REGION
#     )

# # Constants for the dataset
# S3_BUCKET_NAME = os.getenv("BUCKET_NAME")
# YEARS = list(range(2021, 2026))  # 2021 to 2025
# QUARTERS = list(range(1, 5))     # 1 to 4
# CHUNKING_STRATEGIES = ["default", "fixed", "semantic"]

# # Function to load markdown file from S3
# def load_markdown_file_from_s3(s3_client, bucket_name, file_key):
#     try:
#         response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
#         content = response['Body'].read().decode('utf-8')
#         if not content.strip():
#             print(f"Warning: File {file_key} is empty.")
#             return None
#         return content
#     except Exception as e:
#         print(f"Error loading file {file_key}: {e}")
#         return None

# # Chunking strategies with improved semantic chunking
# def chunk_text(text, strategy="default"):
#     if not text:
#         return []
        
#     if strategy == "default":
#         # Split by paragraphs
#         chunks = [s.strip() for s in text.split("\n\n") if s.strip()]
#     elif strategy == "fixed":
#         # Fixed size chunks
#         chunks = [text[i:i+500] for i in range(0, len(text), 500)]
#     elif strategy == "semantic":
#         # Improved semantic chunking - break at sentence boundaries when possible
#         sentences = text.replace("\n", " ").split(". ")
#         chunks = []
#         current_chunk = ""
        
#         for sentence in sentences:
#             if len(current_chunk) + len(sentence) > 300 and current_chunk:
#                 chunks.append(current_chunk.strip())
#                 current_chunk = sentence
#             else:
#                 current_chunk += sentence + ". "
                
#         if current_chunk:
#             chunks.append(current_chunk.strip())
#     else:
#         chunks = [text]
        
#     return [chunk for chunk in chunks if chunk.strip()]  # Ensure no empty chunks

# # Redis key naming functions
# def get_chunks_key(year, quarter, strategy):
#     return f"nvidia:{year}:{quarter}:{strategy}:chunks"

# def get_embeddings_key(year, quarter, strategy):
#     return f"nvidia:{year}:{quarter}:{strategy}:embeddings"

# def get_metadata_key(year, quarter, strategy):
#     return f"nvidia:{year}:{quarter}:{strategy}:metadata"

# # Serialize and deserialize numpy arrays
# def serialize_numpy_array(arr):
#     return pickle.dumps(arr)

# def deserialize_numpy_array(serialized_arr):
#     return pickle.loads(serialized_arr)

# # Store chunks and their embeddings in Redis
# def store_chunks_and_embeddings(year, quarter, strategy, chunks, embeddings):
#     if not chunks or len(chunks) == 0:
#         print(f"Warning: No chunks for {year} Q{quarter} with strategy {strategy}")
#         return False
        
#     chunks_key = get_chunks_key(year, quarter, strategy)
#     embeddings_key = get_embeddings_key(year, quarter, strategy)
#     metadata_key = get_metadata_key(year, quarter, strategy)
    
#     # Store metadata
#     metadata = {
#         "year": year,
#         "quarter": quarter,
#         "strategy": strategy,
#         "num_chunks": len(chunks),
#         "embedding_dim": embeddings.shape[1] if embeddings is not None else 0
#     }
    
#     # Use pipeline for atomic operations
#     pipeline = redis_client.pipeline()
    
#     # Store chunks as a list
#     pipeline.delete(chunks_key)
#     for chunk in chunks:
#         pipeline.rpush(chunks_key, chunk)
    
#     # Store embeddings
#     if embeddings is not None:
#         pipeline.set(embeddings_key, serialize_numpy_array(embeddings))
    
#     # Store metadata
#     pipeline.hmset(metadata_key, metadata)
    
#     # Set expiration (optional, remove if you want data to persist indefinitely)
#     # pipeline.expire(chunks_key, 86400)  # 24 hours
#     # pipeline.expire(embeddings_key, 86400)
#     # pipeline.expire(metadata_key, 86400)
    
#     # Execute all commands
#     pipeline.execute()
#     return True

# # Retrieve chunks and embeddings from Redis
# def get_chunks_and_embeddings(year, quarter, strategy):
#     chunks_key = get_chunks_key(year, quarter, strategy)
#     embeddings_key = get_embeddings_key(year, quarter, strategy)
    
#     # Check if keys exist
#     if not redis_client.exists(chunks_key) or not redis_client.exists(embeddings_key):
#         return None, None
    
#     # Get chunks
#     chunks = redis_client.lrange(chunks_key, 0, -1)
#     chunks = [chunk.decode('utf-8') for chunk in chunks]
    
#     # Get embeddings
#     embeddings_bytes = redis_client.get(embeddings_key)
#     if not embeddings_bytes:
#         return chunks, None
        
#     embeddings = deserialize_numpy_array(embeddings_bytes)
#     return chunks, embeddings

# # Process a single file and store in Redis
# def process_file(s3_client, bucket_name, year, quarter):
#     file_key = f"{year}/{quarter}/mistral/nvidia_{quarter}.md"
#     print(f"Processing {file_key}...")
    
#     markdown_content = load_markdown_file_from_s3(s3_client, bucket_name, file_key)
#     if not markdown_content:
#         print(f"No content found for {file_key}")
#         return False
    
#     success = True
#     for strategy in CHUNKING_STRATEGIES:
#         # Chunk the text
#         chunks = chunk_text(markdown_content, strategy)
#         if not chunks:
#             continue
            
#         # Compute embeddings
#         try:
#             embeddings = manual_model.encode(chunks, convert_to_tensor=True)
#             embeddings_np = embeddings.cpu().numpy()
#         except Exception as e:
#             print(f"Error computing embeddings for {file_key} with strategy {strategy}: {e}")
#             continue
            
#         # Store in Redis
#         if not store_chunks_and_embeddings(year, quarter, strategy, chunks, embeddings_np):
#             success = False
    
#     return success

# # Preprocess all data and store in Redis
# def preprocess_and_store_all_data():
#     try:
#         s3_client = get_s3_client()
        
#         # Check if data is already processed by checking a sample key
#         if redis_client.exists(get_chunks_key(YEARS[0], QUARTERS[0], CHUNKING_STRATEGIES[0])):
#             print("Data already processed in Redis")
#             return True
            
#         for year in YEARS:
#             for quarter in QUARTERS:
#                 success = process_file(s3_client, S3_BUCKET_NAME, year, quarter)
#                 if not success:
#                     print(f"Warning: Failed to process {year} Q{quarter}")
                    
#         return True
#     except Exception as e:
#         print(f"Error preprocessing data: {e}")
#         return False

# # Find similar text chunks
# def find_similar_chunks(query, chunks, embeddings, top_k=3):
#     if not chunks or embeddings is None or len(chunks) == 0:
#         return []
        
#     query_embedding = manual_model.encode(query, convert_to_tensor=True)
#     query_np = query_embedding.cpu().numpy().reshape(1, -1)
    
#     # Calculate cosine similarity
#     similarities = cosine_similarity(query_np, embeddings)[0]
    
#     # Get top_k indices
#     if len(similarities) <= top_k:
#         sorted_indices = np.argsort(similarities)[::-1]
#     else:
#         sorted_indices = np.argsort(similarities)[::-1][:top_k]
    
#     # Get results with similarity scores
#     results = [
#         {"chunk": chunks[idx], "similarity": float(similarities[idx])}
#         for idx in sorted_indices
#     ]
    
#     return results

# # Query data for a specific year, quarter and strategy
# def query_data_single(query, year, quarter, strategy, top_k=3):
#     chunks, embeddings = get_chunks_and_embeddings(year, quarter, strategy)
#     if chunks is None or embeddings is None:
#         return []
        
#     return find_similar_chunks(query, chunks, embeddings, top_k)

# # Combine query results from multiple files
# def query_data_multiple(query, years, quarters, strategy, top_k=3):
#     all_results = []
    
#     for year in years:
#         for quarter in quarters:
#             results = query_data_single(query, year, quarter, strategy, top_k)
#             for result in results:
#                 result["source"] = f"{year} Q{quarter}"
#             all_results.extend(results)
    
#     # Sort by similarity and get top_k overall
#     sorted_results = sorted(all_results, key=lambda x: x["similarity"], reverse=True)
#     return sorted_results[:top_k]

# # Pydantic models for API
# class QueryRequest(BaseModel):
#     query: str
#     years: List[int]
#     quarters: List[int]
#     chunking_strategy: str = "default"
#     top_k: int = 3

# class QueryResponse(BaseModel):
#     query: str
#     matches: List[Dict[str, Any]]
#     total_matches: int

# # Initialize data on startup
# @app.on_event("startup")
# async def startup_event():
#     print("Starting data preprocessing...")
#     preprocess_and_store_all_data()
#     print("Data preprocessing complete.")

# # API endpoint for querying data
# @app.post("/query/", response_model=QueryResponse)
# def query_endpoint(request: QueryRequest):
#     if not request.query.strip():
#         raise HTTPException(status_code=400, detail="Query cannot be empty")
        
#     # Validate years and quarters
#     valid_years = [year for year in request.years if year in YEARS]
#     valid_quarters = [quarter for quarter in request.quarters if quarter in QUARTERS]
    
#     if not valid_years or not valid_quarters:
#         raise HTTPException(status_code=400, detail="Invalid years or quarters")
    
#     if request.chunking_strategy not in CHUNKING_STRATEGIES:
#         raise HTTPException(status_code=400, detail=f"Invalid chunking strategy. Choose from {CHUNKING_STRATEGIES}")
    
#     # Get search results
#     matches = query_data_multiple(
#         request.query, 
#         valid_years, 
#         valid_quarters, 
#         request.chunking_strategy,
#         request.top_k
#     )
    
#     if not matches:
#         return QueryResponse(
#             query=request.query,
#             matches=[],
#             total_matches=0
#         )
        
#     return QueryResponse(
#         query=request.query,
#         matches=matches,
#         total_matches=len(matches)
#     )

# # Health check endpoint
# @app.get("/health")
# def health_check():
#     redis_status = "OK" if redis_client.ping() else "Error"
    
#     # Check if we have at least some data
#     data_status = "Available" if redis_client.exists(
#         get_chunks_key(YEARS[0], QUARTERS[0], CHUNKING_STRATEGIES[0])
#     ) else "Not available"
    
#     return {
#         "status": "healthy",
#         "redis": redis_status,
#         "data": data_status
#     }

# # API endpoint to force reprocessing of data
# @app.post("/reprocess")
# def reprocess_data():
#     # Clear existing data
#     keys = redis_client.keys("nvidia:*")
#     if keys:
#         redis_client.delete(*keys)
        
#     success = preprocess_and_store_all_data()
#     if not success:
#         raise HTTPException(status_code=500, detail="Failed to reprocess data")
        
#     return {"status": "success", "message": "Data reprocessed successfully"}

# # List available data
# @app.get("/data")
# def list_available_data():
#     result = {}
    
#     for year in YEARS:
#         for quarter in QUARTERS:
#             for strategy in CHUNKING_STRATEGIES:
#                 metadata_key = get_metadata_key(year, quarter, strategy)
#                 if redis_client.exists(metadata_key):
#                     metadata = redis_client.hgetall(metadata_key)
#                     if metadata:
#                         processed_metadata = {k.decode('utf-8'): v.decode('utf-8') for k, v in metadata.items()}
#                         result[f"{year}_Q{quarter}_{strategy}"] = processed_metadata
    
#     return {"available_data": result}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("manual_embedding:app", host="0.0.0.0", port=8002, reload=True)






import os
import boto3
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# Initialize FastAPI app
app = FastAPI()

# Load the Sentence Transformer model once for reuse
manual_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize AWS S3 Client
def get_s3_client():
    AWS_ACCESS_KEY_ID = os.getenv("ACCESS_KEY")
    AWS_SECRET_ACCESS_KEY = os.getenv("SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("REGION")
    
    if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION]):
        raise ValueError("AWS credentials not properly set in environment variables")
    
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )

# Constants for the dataset
S3_BUCKET_NAME = os.getenv("BUCKET_NAME")
JSON_BUCKET_NAME = os.getenv("JSON_BUCKET_NAME") # will add these later
YEARS = list(range(2021, 2026))  # 2021 to 2025
QUARTERS = list(range(1, 5))     # 1 to 4
CHUNKING_STRATEGIES = ["default", "fixed", "semantic"]

# Function to load markdown file from S3
def load_markdown_file_from_s3(s3_client, bucket_name, file_key):
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        content = response['Body'].read().decode('utf-8')
        if not content.strip():
            print(f"Warning: File {file_key} is empty.")
            return None
        return content
    except Exception as e:
        print(f"Error loading file {file_key}: {e}")
        return None

# Chunking strategies with improved semantic chunking
def chunk_text(text, strategy="default"):
    if not text:
        return []
        
    if strategy == "default":
        # Split by paragraphs
        chunks = [s.strip() for s in text.split("\n\n") if s.strip()]
    elif strategy == "fixed":
        # Fixed size chunks
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    elif strategy == "semantic":
        # Improved semantic chunking - break at sentence boundaries when possible
        sentences = text.replace("\n", " ").split(". ")
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > 300 and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += sentence + ". "
                
        if current_chunk:
            chunks.append(current_chunk.strip())
    else:
        chunks = [text]
        
    return [chunk for chunk in chunks if chunk.strip()]  

# Store chunks and their embeddings in S3
def store_chunks_and_embeddings(year, quarter, strategy, chunks, embeddings):
    if not chunks or len(chunks) == 0:
        print(f"Warning: No chunks for {year} Q{quarter} with strategy {strategy}")
        return False
        
    # Create JSON data
    data = {
        "year": year,
        "quarter": quarter,
        "strategy": strategy,
        "chunks": chunks,
        "embeddings": embeddings.tolist()  # numpy array to list 
    }
    
    # Store JSON in S3
    s3_client = get_s3_client()
    file_key = f"{year}/{quarter}/{strategy}.json"
    try:
        s3_client.put_object(Body=json.dumps(data), Bucket=JSON_BUCKET_NAME, Key=file_key)
        return True
    except Exception as e:
        print(f"Error storing data in S3: {e}")
        return False

# Retrieve chunks and embeddings from S3
def get_chunks_and_embeddings(year, quarter, strategy):
    s3_client = get_s3_client()
    file_key = f"{year}/{quarter}/{strategy}.json"
    
    try:
        response = s3_client.get_object(Bucket=JSON_BUCKET_NAME, Key=file_key)
        data = json.loads(response['Body'].read())
        chunks = data['chunks']
        embeddings = np.array(data['embeddings'])  # Convert list back to numpy array
        return chunks, embeddings
    except Exception as e:
        print(f"Error loading data from S3: {e}")
        return None, None

# Process a single file and store in S3
def process_file(s3_client, bucket_name, year, quarter):
    file_key = f"{year}/{quarter}/mistral/nvidia_{quarter}.md"
    print(f"Processing {file_key}...")
    
    markdown_content = load_markdown_file_from_s3(s3_client, bucket_name, file_key)
    if not markdown_content:
        print(f"No content found for {file_key}")
        return False
    
    success = True
    for strategy in CHUNKING_STRATEGIES:
        # Chunk the text
        chunks = chunk_text(markdown_content, strategy)
        if not chunks:
            continue
            
        # Compute embeddings
        try:
            embeddings = manual_model.encode(chunks, convert_to_tensor=True)
            embeddings_np = embeddings.cpu().numpy()
        except Exception as e:
            print(f"Error computing embeddings for {file_key} with strategy {strategy}: {e}")
            continue
            
        # Store in S3
        if not store_chunks_and_embeddings(year, quarter, strategy, chunks, embeddings_np):
            success = False
    
    return success

# Preprocess all data and store in S3
def preprocess_and_store_all_data():
    try:
        s3_client = get_s3_client()
        
        for year in YEARS:
            for quarter in QUARTERS:
                success = process_file(s3_client, S3_BUCKET_NAME, year, quarter)
                if not success:
                    print(f"Warning: Failed to process {year} Q{quarter}")
                    
        return True
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return False

# Find similar text chunks
def find_similar_chunks(query, chunks, embeddings, top_k=3):
    if not chunks or embeddings is None or len(chunks) == 0:
        return []
        
    query_embedding = manual_model.encode(query, convert_to_tensor=True)
    query_np = query_embedding.cpu().numpy().reshape(1, -1)
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_np, embeddings)[0]
    
    # Get top_k indices
    if len(similarities) <= top_k:
        sorted_indices = np.argsort(similarities)[::-1]
    else:
        sorted_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Get results with similarity scores
    results = [
        {"chunk": chunks[idx], "similarity": float(similarities[idx])}
        for idx in sorted_indices
    ]
    
    return results

# Query data for a specific year, quarter and strategy
def query_data_single(query, year, quarter, strategy, top_k=3):
    chunks, embeddings = get_chunks_and_embeddings(year, quarter, strategy)
    if chunks is None or embeddings is None:
        return []
        
    return find_similar_chunks(query, chunks, embeddings, top_k)

# Combine query results from multiple files
def query_data_multiple(query, years, quarters, strategy, top_k=3):
    all_results = []
    
    for year in years:
        for quarter in quarters:
            results = query_data_single(query, year, quarter, strategy, top_k)
            for result in results:
                result["source"] = f"{year} Q{quarter}"
            all_results.extend(results)
    
    # Sort by similarity and get top_k overall
    sorted_results = sorted(all_results, key=lambda x: x["similarity"], reverse=True)
    return sorted_results[:top_k]

# Pydantic models for API
class QueryRequest(BaseModel):
    query: str
    years: List[int]
    quarters: List[int]
    chunking_strategy: str = "default"
    top_k: int = 3

class QueryResponse(BaseModel):
    query: str
    matches: List[Dict[str, Any]]
    total_matches: int

# Initialize data on startup
@app.on_event("startup")
async def startup_event():
    print("Starting data preprocessing...")
    preprocess_and_store_all_data()
    print("Data preprocessing complete.")

# API endpoint for querying data
@app.post("/query/", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    # Validate years and quarters
    valid_years = [year for year in request.years if year in YEARS]
    valid_quarters = [quarter for quarter in request.quarters if quarter in QUARTERS]
    
    if not valid_years or not valid_quarters:
        raise HTTPException(status_code=400, detail="Invalid years or quarters")
    
    if request.chunking_strategy not in CHUNKING_STRATEGIES:
        raise HTTPException(status_code=400, detail=f"Invalid chunking strategy. Choose from {CHUNKING_STRATEGIES}")
    
    # Get search results
    matches = query_data_multiple(
        request.query, 
        valid_years, 
        valid_quarters, 
        request.chunking_strategy,
        request.top_k
    )
    
    if not matches:
        return QueryResponse(
            query=request.query,
            matches=[],
            total_matches=0
        )
        
    return QueryResponse(
        query=request.query,
        matches=matches,
        total_matches=len(matches)
    )

# Health check endpoint
@app.get("/health")
def health_check():
    s3_client = get_s3_client()
    file_key = f"{YEARS[0]}/{QUARTERS[0]}/{CHUNKING_STRATEGIES[0]}.json"
    data_status = "Available" if s3_client.list_objects_v2(Bucket=JSON_BUCKET_NAME, Prefix=file_key) else "Not available"
    
    return {
        "status": "healthy",
        "data": data_status
    }

# API endpoint to force reprocessing of data
@app.post("/reprocess")
def reprocess_data():
    # Clear existing data
    s3_client = get_s3_client()
    objects_to_delete = []
    for year in YEARS:
        for quarter in QUARTERS:
            for strategy in CHUNKING_STRATEGIES:
                file_key = f"{year}/{quarter}/{strategy}.json"
                objects_to_delete.append({"Key": file_key})
                
    if objects_to_delete:
        s3_client.delete_objects(Bucket=JSON_BUCKET_NAME, Delete={"Objects": objects_to_delete})
        
    success = preprocess_and_store_all_data()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to reprocess data")
        
    return {"status": "success", "message": "Data reprocessed successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("manual_embedding:app", host="0.0.0.0", port=8002, reload=True)

