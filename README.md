# AI_Powered_Information_Retrieval

## Streamlit : https://pytract-rag.streamlit.app/
## Api = https://rag-798800248787.us-central1.run.app

# Data Flow for Data Extraction and Querying System

This repository provides a comprehensive system for extracting, processing, storing, and querying data. Below is a detailed explanation of the system architecture, as illustrated in the diagram.

---

## **System Overview**

The system consists of four key components:

1. **Data Extraction**
   - **Tools Used**: Selenium for web scraping.
   - **Processing**: Extracted data is processed using Docling and Mistralai for document parsing and preparation.

2. **Data Storage**
   - **Storage Medium**: AWS S3 bucket is used for scalable and centralized storage of processed data.

3. **Data Processing**
   - Three pipelines are supported:
     - **Pinecone**: Handles chunking strategies and embeddings generation.
     - **ChromaDB**: Offers an alternative processing pipeline with similar functionality.
     - **Manual DB**: Allows users to define custom chunking strategies and embeddings generation.

4. **Data Querying**
   - **Backend**: FastAPI serves as the querying interface.
   - **Frontend**: Streamlit UI provides a user-friendly interface for interacting with the processed data.
   - 
## **Architecture Diagram**

Below is the architecture diagram illustrating the data flow:

![Data Flow Diagram](https://pplx-res.cloudinary.com/image/upload/v1742583192/user_uploads/QifyRUrvYHUonpS/image.jpg)

### **Key Components in the Diagram**

1. **Data Extraction**
   - Data scraping is performed using Selenium.
   - Data is parsed and prepared using Docling & Mistralai.

2. **Data Storage**
   - The processed data is stored in an AWS S3 bucket for centralized access.

3. **Data Processing Pipelines**
   - **Pinecone Pipeline**:
     - Implements chunking strategies.
     - Generates embeddings for efficient querying.
   - **ChromaDB Pipeline**:
     - Similar to Pinecone but uses ChromaDB for storage and querying.
   - **Manual DB Pipeline**:
     - Provides a customizable option for chunking and embeddings generation.

4. **Data Querying**
   - FastAPI serves as the backend API for querying.
   - Streamlit UI provides a simple front-end interface for users to interact with the system.

---

## **Features**

- Scalable data storage using AWS S3.
- Flexible processing pipelines (Pinecone, ChromaDB, Manual DB).
- User-friendly querying interface with FastAPI and Streamlit UI.
- Customizable chunking strategies for tailored data processing.

---

## **Setup Instructions**

### Prerequisites

- Python 3.8+
- AWS CLI configured with access to an S3 bucket
- Selenium WebDriver installed
- Docker (optional, for containerized deployment)

### Installation

1. Clone this repository:
  
2. Install dependencies:


3. Configure your environment variables:
Create a `.env` file in the root directory with the following keys:

AWS_ACCESS_KEY

AWS_SECRET_KEY

S3_BUCKET_NAME



4. Set up Selenium WebDriver:
Download the appropriate WebDriver for your browser and add it to your system's PATH.

5. Start the API and UI:

uvicorn main:app --reload # For FastAPI backend

streamlit run main.py # For Streamlit UI
