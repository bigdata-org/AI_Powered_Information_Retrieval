import streamlit as st
import requests
import json

# FastAPI backend URL
BASE_URL = "http://127.0.0.1:8000"  # Update with deployed backend URL if needed

st.title("📊 NVIDIA Financial Reports RAG System")
st.sidebar.header("⚙️ User Settings")

# Year and Quarter Selection
years = st.sidebar.multiselect("Select Years", [2020, 2021, 2022, 2023, 2024])
quarters = st.sidebar.multiselect("Select Quarters", [1, 2, 3, 4])

# Chunking Strategy Selection
chunking_strategy = st.sidebar.radio("Select Chunking Strategy", [
    "Paragraph-Based", "Fixed-Length", "Semantic"
])

# User Query Input
query = st.text_area("🔍 Enter Your Query", "Compare Q1 2024 with Q2 2025")

# Fetch Data Button
if st.button("🔎 Get Answer"):
    if not years or not quarters:
        st.warning("⚠️ Please select at least one year and one quarter.")
    else:
        st.info("⏳ Fetching response... Please wait.")
        
        # Define payload
        payload = {
            "query": query,
            "years": years,
            "quarters": quarters,
            "chunking_strategy": chunking_strategy
        }
        
        try:
            response = requests.post(f"{BASE_URL}/query/", json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success("✅ Response Generated!")
                st.write("### 📌 Answer:")
                st.write(result["response"])
            else:
                st.error(f"❌ Error: {response.json()['detail']}")
        except Exception as e:
            st.error(f"❌ Failed to connect to backend: {str(e)}")

st.sidebar.write("🖥️ Powered by FastAPI & Streamlit 🚀")
