import streamlit as st
import requests

# Backend FastAPI URL
BACKEND_URL = "https://rag-798800248787.us-central1.run.app"  # Adjust as needed
# st.session_state['metadata'] = requests.get('https://rag-pipeline-data.s3.us-east-2.amazonaws.com/metadata/metadata_s3url.json').json()
model_mapper = {
    "openai/gpt-4o": "gpt-4o-2024-08-06",
    "openai/gpt-3.5-turbo": "gpt-3.5-turbo-0125",
    "openai/gpt-4": "gpt-4-0613",
    "openai/gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gemini/gemini-1.5-pro": "gemini/gemini-1.5-pro",
    "gemini/gemini-2.0-flash-lite": "gemini/gemini-2.0-flash-lite"
}

available_models = list(model_mapper.keys())
chunking_strategies = ['sentence-5', 'word-400-overlap-40', 'char-1200-overlap-120']
# Function to call the query API
def rag(p_url, model, prompt, chunking_strategy, db, search_params, mode):
    url = f"{BACKEND_URL}/qa"
    data = {
            "url": p_url, 
            "model": model_mapper[model],
            "mode": mode,
            "prompt": prompt,
            "chunking_strategy": chunking_strategy,
            "db": db,
            "search_params": search_params
    }
    response = requests.post(url, json=data)
    return response.json()

def upload(uploaded_file):
    url = f"{BACKEND_URL}/upload_pdf"
    files = {"file": uploaded_file}
    response = requests.post(url, files=files)
    return response.json()['url'] #return array of 2 links

def index(p_url, db, chunking_strategy):
    url = f"{BACKEND_URL}/index"
    data = {
            "url": p_url, 
            "chunking_strategy": chunking_strategy,
            "db": db
    }
    response = requests.post(url, json=data)
    return response.json() #return array of 2 links 

# Streamlit UI
st.title("AI-powered Query System")

# Sidebar Navigation
page = st.sidebar.selectbox("Select Page", ["Financial Report Query", "PDF Query"])

if page == "Financial Report Query":
    mode='nvidia'
    st.header("Financial Report Query")
    
    # Year-Quarter selection dropdown
    year_quarters = [f"{year}_Q{q}" for year in range(2021, 2026) for q in range(1, 5)]
    selected_year_quarters = st.multiselect("Select Year-Quarter Combinations:", year_quarters, default=["2025_Q1"])
    
    # Convert selected year-quarter into required format
    search_params = [{"year": yq.split("_Q")[0], "qtr": yq.split("_Q")[1]} for yq in selected_year_quarters]
    
    # User input for query
    query = st.text_area("Enter your query:", "", height=150)
    
    # Database choice
    db_choice = st.selectbox("Select the data source:", ["pinecone", "chromadb", "manual"])
    model_choice = st.selectbox("Select the data source:", available_models)
    chunking_strategy = st.selectbox("Select chunking strategy:", chunking_strategies)

    # Button to submit the query
    if st.button("Submit Query"):
        if query.strip():
            with st.spinner("Querying the backend..."):
                result = rag(None, model_choice, query, chunking_strategy, db_choice, search_params, 'nvidia')
            
            if "markdown" in result:
                st.subheader("Generated Response:")
                st.write(result["markdown"])
            else:
                st.error("Something went wrong")
        else:
            st.error("Please enter a valid query.")

elif page == "PDF Query":
    st.header("Upload and Query PDF")
    mode='custom'
    # File uploader for PDF
    uploaded_pdf = st.file_uploader("Upload PDF File", type=["pdf"])
    
    # Model choice dropdown
    tool_choice = st.selectbox("Select response model:", ["mistral", "docling"])
    db_choice = st.selectbox("Select the data source:", ["pinecone", "chromadb", "manual"])
    # Chunking strategy
    chunking_strategy = st.selectbox("Select chunking strategy:", chunking_strategies)
    model_choice = st.selectbox("Select the data source:", available_models)
    # User input for query
    query = st.text_area("Enter your query:", "", height=150)
    
    # Button to process PDF
    if st.button("Submit Query"):
        if uploaded_pdf and query.strip():
            with st.spinner("Processing PDF and querying..."):
                endpoints = upload(uploaded_pdf)
                md_endpoint = endpoints[0] if tool_choice == 'docling' else endpoints[1]
                index_response = index(md_endpoint, db_choice, chunking_strategy)
                result = rag(md_endpoint, model_choice, query, chunking_strategy, db_choice, [{"src":md_endpoint}], 'custom')
            if "markdown" in result:
                st.subheader("Generated Response:")
                st.write(result["markdown"])
            else:
                st.error("Error: Could not process PDF.")
        else:
            st.error("Please upload a PDF and enter a query.")
