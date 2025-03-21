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

Use FastAPI to connect the RAG pipeline and return relevant document chunks based on user queries. Leverage your preferred LLM to process and generate responses.

### **6. Deployment**
Create two Docker pipelines:
1. **Airflow pipeline**: For data ingestion, processing, and retrieval.
2. **Streamlit + FastAPI pipeline**: For user interaction and querying.

### README.md
Include comprehensive instructions for project setup and usage:
1. How to clone the repository.
2. Steps to set up Docker pipelines.
3. How to run the Airflow pipeline.
4. Instructions for using the Streamlit application.
5. Explanation of available features (PDF upload, parser selection, RAG methods, etc.).

### AIUseDisclosure.md
Detail all AI tools used in the project and their purpose.


---

## Links to GitHub Tasks
Each team member must document their contributions clearly by linking tasks owned by them within the repository's issue tracker.
