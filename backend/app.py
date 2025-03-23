from fastapi import FastAPI, HTTPException, UploadFile, File
from utils.aws.s3 import *
from utils.docling.core import docling_PDF2MD
from utils.mistral.core import mistral_parser as mistral_PDF2MD
from utils.snowflake.core import sf_litellm_read, sf_litellm_write
from utils.pytract.pytract_db import pytract_db
from utils.pytract.pytract_rag import pytract_rag
from pydantic import BaseModel
from typing import List
from io import BytesIO
from  dotenv import load_dotenv
import redis
import logging
from typing import List, Dict, Any, Optional
from utils.helper import *
from utils.litellm.llm import llm

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)
load_dotenv()

class qaModel(BaseModel):
    url: Optional[str] = None
    model: str
    mode: str
    prompt: str
    chunking_strategy: str
    db: str
    search_params: List[Dict[str, Any]]
    
class UrlModel(BaseModel):
    url: str
    
class indexModel(BaseModel):
    url: str
    db: str
    chunking_strategy: str
    
app = FastAPI()

# Upload PDF endpoint
@app.post('/upload_pdf')
async def upload_pdf(file: UploadFile = File(...)) :
    try:
        file_content = await file.read()
        file_name = file.filename.split("/")[-1].split(".")[0]
        if not is_file_size_within_limit(file_content):
            handle_invalid_file_size()
        try:
            file_bytes_io = BytesIO(file_content)
        except Exception as e:
            handle_invalid_pdf()
        s3_client = get_s3_client()
        if s3_client == -1:
            handle_s3_error()
        endpoint = upload_pdf_to_s3(s3_client, file_name, file_bytes_io)
        if endpoint == -1:
            handle_internal_server_error()
        docling_md_endpoint = docling_PDF2MD(endpoint)
        mistral_md_endpoint = mistral_PDF2MD(endpoint)
        return {"url": [docling_md_endpoint, mistral_md_endpoint]}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise e

@app.post('/index')
async def index_md(request: indexModel):
    url = request.url
    db = request.db
    chunking_strategy = request.chunking_strategy
    mode='custom'
    if db in ["pinecone", "chromadb"]:
        rag = pytract_rag(mode, db, chunking_strategy)
        rag.run_custom_indexing_pipeline(url)
    else:
        rag = pytract_db(chunking_strategy=chunking_strategy)
        rag.run_indexing_pipeline(url)
        redis_key = f"{url}:{chunking_strategy}"
        num_embeddings = rag._get_embeddings(redis_key)
    return {"status":200, "detail":"ok"}

@app.post('/qa') 
async def qa_pipeline(request: qaModel):
    try:
        url = request.url
        model = request.model
        prompt = request.prompt
        chunking_strategy = request.chunking_strategy
        mode = request.mode
        db = request.db
        search_params = request.search_params
        logger.info(f"URL: {url}, Model: {model}, Prompt: {prompt}, Chunking Strategy: {chunking_strategy}, Mode: {mode}, DB: {db}, Search Params: {search_params}")
        if invalid_model(model):
            raise handle_invalid_model()
        if invalid_prompt(prompt):
            raise handle_invalid_prompt()
        
        if db in ['pinecone', 'chromadb'] and mode=='nvidia': 
            rag = pytract_rag(mode, db, chunking_strategy)
            response = rag.run_nvidia_text_generation_pipeline(search_params, prompt, model) 
        elif db in ['pinecone', 'chromadb'] and mode=='custom': 
            rag = pytract_rag(mode, db, chunking_strategy)
            response = rag.run_custom_text_generation_pipeline(search_params[0]['src'], prompt, model)
        elif db=='manual' and mode=='custom':
            rag = pytract_db(chunking_strategy=chunking_strategy)
            redis_key=f"{url}:{chunking_strategy}"
            context = rag.get_top_n_chunks(redis_key, prompt, n=5)
            response = llm(model=model, prompt=context)
        else:
            response={}
            response['markdown'] = 'UI Cutomization is invalid'
        response['mode'] = 'qa'
        response['source'] = url
        # sf_litellm_write(response)
        return {"markdown": response['markdown']}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise e