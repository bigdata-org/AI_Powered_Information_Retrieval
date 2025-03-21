# import streamlit as st
# import requests

# # Backend FastAPI URL
# BACKEND_URL = "http://localhost:8000"  # Adjust as needed

# # Function to call the query API
# def query_backend(query, search_params, db_choice, model_choice, chunking_strategy):
#     url = f"{BACKEND_URL}/query/"
#     data = {
#         "query": query,
#         "search_params": search_params,
#         "db_choice": db_choice,
#         "model_choice": model_choice,
#         "chunking_strategy": chunking_strategy,
#     }
#     response = requests.post(url, json=data)
#     return response.json()

# # Function to call the PDF query API
# def query_pdf(pdf_file, query, model_choice, chunking_strategy):
#     url = f"{BACKEND_URL}/query_pdf/"
#     files = {"pdf_file": pdf_file}
#     data = {
#         "query": query,
#         "model_choice": model_choice,
#         "chunking_strategy": chunking_strategy,
#     }
#     response = requests.post(url, files=files, data=data)
#     return response.json()

# # Streamlit UI
# st.title("AI-powered Query System")

# # Sidebar Navigation
# page = st.sidebar.selectbox("Select Page", ["Financial Report Query", "PDF Query"])

# if page == "Financial Report Query":
#     st.header("Financial Report Query")
    
#     # Year-Quarter selection dropdown
#     year_quarters = [f"{year}_Q{q}" for year in range(2021, 2026) for q in range(1, 5)]
#     selected_year_quarters = st.multiselect("Select Year-Quarter Combinations:", year_quarters, default=["2024_Q1"])
    
#     # Convert selected year-quarter into required format
#     search_params = [{"year": yq.split("_Q")[0], "qtr": yq.split("_Q")[1]} for yq in selected_year_quarters]
    
#     # User input for query
#     query = st.text_area("Enter your query:", "", height=150)
    
#     # Database choice
#     db_choice = st.selectbox("Select the data source:", ["pinecone", "chromadb", "manual"])
    
#     # Chunking strategy
#     chunking_strategy = st.selectbox("Select chunking strategy:", ["default", "fixed", "semantic"])
    
#     # Model choice (only mistral)
#     model_choice = "mistral"
    
#     # Button to submit the query
#     if st.button("Submit Query"):
#         if query.strip():
#             with st.spinner("Querying the backend..."):
#                 result = query_backend(query, search_params, db_choice, model_choice, chunking_strategy)
            
#             if "response" in result:
#                 st.subheader("Generated Response:")
#                 st.write(result["response"])
#             else:
#                 st.error("Error: Could not fetch data.")
#         else:
#             st.error("Please enter a valid query.")

# elif page == "PDF Query":
#     st.header("Upload and Query PDF")
    
#     # File uploader for PDF
#     uploaded_pdf = st.file_uploader("Upload PDF File", type=["pdf"])
    
#     # Model choice dropdown
#     model_choice = st.selectbox("Select response model:", ["mistral", "docling"])
    
#     # Chunking strategy
#     chunking_strategy = st.selectbox("Select chunking strategy:", ["default", "fixed", "semantic"])
    
#     # User input for query
#     query = st.text_area("Enter your query:", "", height=150)
    
#     # Button to process PDF
#     if st.button("Submit Query"):
#         if uploaded_pdf and query.strip():
#             with st.spinner("Processing PDF and querying..."):
#                 result = query_pdf(uploaded_pdf, query, model_choice, chunking_strategy)
            
#             if "response" in result:
#                 st.subheader("Generated Response:")
#                 st.write(result["response"])
#             else:
#                 st.error("Error: Could not process PDF.")
#         else:
#             st.error("Please upload a PDF and enter a query.")







import streamlit as st
import requests
import time

# Backend FastAPI URL
BACKEND_URL = "http://localhost:8000"  # Adjust as needed

# Function to call the query API
def query_backend(query, search_params, db_choice, model_choice, chunking_strategy):
    url = f"{BACKEND_URL}/query/"
    data = {
        "query": query,
        "search_params": search_params,
        "db_choice": db_choice,
        "model_choice": model_choice,
        "chunking_strategy": chunking_strategy,
    }
    response = requests.post(url, json=data)
    return response.json()

# Function to call the PDF query API
def query_pdf(pdf_file, query, model_choice, chunking_strategy):
    url = f"{BACKEND_URL}/query_pdf/"
    files = {"pdf_file": pdf_file}
    data = {
        "query": query,
        "model_choice": model_choice,
        "chunking_strategy": chunking_strategy,
    }
    response = requests.post(url, files=files, data=data)
    return response.json()

# Streamlit UI
st.title("AI-powered Query System")

# Sidebar Navigation
page = st.sidebar.selectbox("Select Page", ["Financial Report Query", "PDF Query"])

if page == "Financial Report Query":
    st.header("Financial Report Query")
    
    # Year-Quarter selection dropdown
    year_quarters = [f"{year}_Q{q}" for year in range(2021, 2026) for q in range(1, 5)]
    selected_year_quarters = st.multiselect("Select Year-Quarter Combinations:", year_quarters, default=["2024_Q1"])
    
    # Convert selected year-quarter into required format
    search_params = [{"year": yq.split("_Q")[0], "qtr": yq.split("_Q")[1]} for yq in selected_year_quarters]
    
    # User input for query
    query = st.text_area("Enter your query:", "", height=150)
    
    # Database choice
    db_choice = st.selectbox("Select the data source:", ["pinecone", "chromadb", "manual"])
    
    # Chunking strategy
    chunking_strategy = st.selectbox("Select chunking strategy:", ["default", "fixed", "semantic"])
    
    # Model choice (only mistral)
    model_choice = "mistral"
    
    # Button to submit the query
    if st.button("Submit Query"):
        if query.strip():
            with st.spinner("Querying the backend..."):
                result = query_backend(query, search_params, db_choice, model_choice, chunking_strategy)
            
            if "response" in result:
                st.subheader("Generated Response:")
                st.write(result["response"])
            else:
                st.error("Error: Could not fetch data.")
        else:
            st.error("Please enter a valid query.")

elif page == "PDF Query":
    st.header("Upload and Query PDF")
    
    # Create a placeholder for the progress bar
    progress_placeholder = st.empty()
    progress_bar = None
    status_text = st.empty()
    
    # Initialize session state for tracking progress
    if 'progress_state' not in st.session_state:
        st.session_state.progress_state = 0
        st.session_state.process_started = False
        st.session_state.pdf_uploaded = False
        st.session_state.model_selected = False
        st.session_state.chunking_selected = False
        st.session_state.query_entered = False
    
    # File uploader for PDF
    uploaded_pdf = st.file_uploader("Upload PDF File", type=["pdf"])
    
    # Update progress when PDF is uploaded
    if uploaded_pdf and not st.session_state.pdf_uploaded:
        st.session_state.pdf_uploaded = True
        st.session_state.progress_state = 0.25
        
        # Display progress bar when process starts
        if not st.session_state.process_started:
            progress_bar = progress_placeholder.progress(st.session_state.progress_state)
            status_text.text("PDF uploaded. Processing...")
            time.sleep(10)  # Sleep for 10 seconds
            st.session_state.process_started = True
    
    # Model choice dropdown
    model_choice = st.selectbox("Select response model:", ["mistral", "docling"])
    
    # Update progress when model is selected
    if model_choice and not st.session_state.model_selected:
        st.session_state.model_selected = True
        st.session_state.progress_state = 0.5
        if st.session_state.process_started:
            progress_bar = progress_placeholder.progress(st.session_state.progress_state)
            status_text.text("Model selected. Preparing chunking options...")
            time.sleep(10)  # Sleep for 10 seconds
    
    # Chunking strategy
    chunking_strategy = st.selectbox("Select chunking strategy:", ["default", "fixed", "semantic"])
    
    # Update progress when chunking strategy is selected
    if chunking_strategy and not st.session_state.chunking_selected:
        st.session_state.chunking_selected = True
        st.session_state.progress_state = 0.75
        if st.session_state.process_started:
            progress_bar = progress_placeholder.progress(st.session_state.progress_state)
            status_text.text("Chunking strategy selected. Ready for query...")
            time.sleep(10)  # Sleep for 10 seconds
    
    # User input for query
    query = st.text_area("Enter your query:", "", height=150)
    
    # Update progress when query is entered
    if query.strip() and not st.session_state.query_entered:
        st.session_state.query_entered = True
        st.session_state.progress_state = 0.9
        if st.session_state.process_started:
            progress_bar = progress_placeholder.progress(st.session_state.progress_state)
            status_text.text("Query entered. Ready to submit...")
            time.sleep(10)  # Sleep for 10 seconds
    
    # Button to process PDF
    if st.button("Submit Query"):
        if uploaded_pdf and query.strip():
            st.session_state.progress_state = 1.0
            progress_bar = progress_placeholder.progress(st.session_state.progress_state)
            status_text.text("Processing query...")
            
            with st.spinner("Processing PDF and querying..."):
                # Sleep for 10 seconds to simulate processing
                time.sleep(10)
                result = query_pdf(uploaded_pdf, query, model_choice, chunking_strategy)
            
            status_text.text("Processing complete!")
            
            if "response" in result:
                st.subheader("Generated Response:")
                st.write(result["response"])
            else:
                st.error("Error: Could not process PDF.")
                
            # Reset progress state for next query
            st.session_state.progress_state = 0
            st.session_state.process_started = False
            st.session_state.pdf_uploaded = False
            st.session_state.model_selected = False
            st.session_state.chunking_selected = False
            st.session_state.query_entered = False
        else:
            st.error("Please upload a PDF and enter a query.")