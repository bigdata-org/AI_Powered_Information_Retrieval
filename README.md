# AI_Powered_Information_Retrieval

## Streamlit : https://pytract-rag.streamlit.app/
## Api = https://rag-798800248787.us-central1.run.app

## **Overview**

This involves designing and implementing a Retrieval-Augmented Generation (RAG) pipeline using Apache Airflow for orchestrating workflows. The goal is to build an AI-powered information retrieval application that processes unstructured data sources such as PDFs and web pages. The pipeline must be modular and extensible for future applications.
## **Requirements**

### **1. Data**
- Collect NVIDIA quarterly reports for the past five years.

### **2. Data Pipeline**
- Use Apache Airflow to orchestrate workflows for:
  - Data ingestion
  - Processing
  - Retrieval

### **3. PDF Parsing Strategies**
Implement three strategies for parsing PDFs:
1. Build upon Assignment 1’s extraction capabilities.
2. Use Docling for parsing PDFs.
3. Explore Mistral OCR for improved text extraction.

### **4. RAG Pipeline Implementation**
- Implement a naive RAG system without a vector database, computing embeddings and cosine similarity manually.
- Integrate with Pinecone for vector-based retrieval.
- Integrate with ChromaDB for advanced retrieval.
- Implement at least three chunking strategies to optimize retrieval.
- Enable hybrid search to query specific quarter data for context.

### **5. Testing & User Interface**
Develop a Streamlit application with the following features:
- Upload PDFs.
- Select PDF parser (Docling, Mistral OCR, etc.).
- Choose RAG method (manual embeddings, Pinecone, ChromaDB).
- Select chunking strategy.
- Query specific quarter/quarters data to retrieve context.

Used FastAPI to connect the RAG pipeline and return relevant document chunks based on user queries. Leverage your LLM to process and generate responses.

### **6. Deployment**
Create two Docker pipelines:
1. **Airflow pipeline**: For data ingestion, processing, and retrieval.
2. **Streamlit + FastAPI pipeline**: For user interaction and querying.


Detail all tools used in the project and their purpose.
References
Apache Airflow Documentation – Workflow orchestration and automation.
🔗https://airflow.apache.org/docs/

Docling GitHub Repository – PDF parsing and text extraction.
🔗https://github.com/docling-ai

Mistral OCR – AI-powered OCR for improved text extraction.
🔗https://mistral.ai/news/mistral-ocr

Pinecone Documentation – Vector database for fast similarity search.
🔗https://docs.pinecone.io/

ChromaDB Documentation – Open-source vector database for AI applications.
🔗https://docs.trychroma.com/

FastAPI Documentation – High-performance API framework for Python.
🔗https://fastapi.tiangolo.com/

Streamlit Documentation – Interactive UI framework for data applications.
🔗https://docs.streamlit.io/

TF-IDF & Cosine Similarity – Methods for text similarity and retrieval.
🔗https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting

Docker Documentation – Containerization for deployment.
🔗https://docs.docker.com/

These references provide the foundational knowledge and tools used in implementing the AI-powered RAG pipeline for information retrieval. 🚀

AI-assisted tools were utilized to enhance the development process, including:

Code optimization and debugging using AI-based suggestions.

Documentation generation and structuring for better clarity.

Automating repetitive coding tasks to improve efficiency.

Reference content and functionality insights were derived using ChatGPT, Perplexity, and DeepSeek.

