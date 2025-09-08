__import__("pysqlite3")
import sys
sys.modules["sqlite3"]=sys.modules.pop("pysqlite3")

import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import os
import tempfile
import shutil
from dotenv import load_dotenv
import atexit
import re

# Page Config FIRST
st.set_page_config(page_title="Research Assistant", layout="wide")

# Load environment variables
load_dotenv()

# API Key Handling
groq_api = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
if not groq_api:
    st.error("GROQ API Key not found! Set it in Streamlit Secrets or .env file.")
    st.stop()

# Custom CSS for better UI
st.markdown("""
<style>
    body {font-family: 'Segoe UI', sans-serif;}
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 6px;
        font-weight: bold;
        padding: 8px 16px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.03);
    }
    .url-input {
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

def remove_think_tags(text):
    cleaned_text = re.sub(r'<(think|thinking|reflection)>.*?</\1>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()

# Sidebar: URL Input Section
st.sidebar.title("Settings")
st.sidebar.write("📑 Enter up to 5 article URLs:")
urls = []
for i in range(5):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}", placeholder="https://example.com/article")
    if url:
        urls.append(url)
st.sidebar.write("🤖 Enter your favourite GROQ model (or leave blank for default [qwen/qwen3-32b]):")
groq_model=st.sidebar.text_input("E.g.: llama-3.3-70b-versatile or llama-3.1-8b-instant") or "qwen/qwen3-32b"

# Process Button in Sidebar
if st.sidebar.button("Build Knowledge Base") and urls:
    with st.spinner("Processing articles and creating embeddings..."):
        documents = []
        for url in urls:
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, "html.parser")
                for script in soup(["script", "style"]):
                    script.decompose()
                paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
                text = " ".join(paragraphs)
                if text:
                    documents.append(Document(page_content=text, metadata={"source": url}))
            except Exception as e:
                st.error(f"Error processing {url}: {e}")
        if documents:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)

            if "embeddings" not in st.session_state:
                st.session_state.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    cache_folder="models"
                )
            temp_dir = tempfile.mkdtemp()
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=st.session_state.embeddings,
                persist_directory=temp_dir
            )
            st.session_state.vectorstore = vectorstore
            st.session_state.temp_dir = temp_dir
            st.session_state.llm = ChatGroq(model=groq_model, groq_api_key=groq_api)
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=st.session_state.llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )
            st.sidebar.success(f"✅ Knowledge base created from {len(documents)} articles!")

# Main Chatbot Area
st.title("Research Assistant")
st.write("Ask questions based on the knowledge base built from your articles.")
if "qa_chain" in st.session_state:
    question = st.chat_input("Enter your question (press Enter to submit)")
    if question:
        with st.spinner("Searching for the best answer..."):
            result = st.session_state.qa_chain.invoke({"query": question})
            st.markdown("### Answer:")
            st.write(remove_think_tags(result["result"]))
            with st.expander("Sources"):
                sources = {doc.metadata["source"] for doc in result["source_documents"]}
                for i, source in enumerate(sources, 1):
                    st.write(f"{i}. {source}")
else:
    st.info("Add some URLs in the sidebar and click **Build Knowledge Base** to start.")

# Cleanup function for temporary directory
def cleanup():
    if "temp_dir" in st.session_state and os.path.exists(st.session_state.temp_dir):
        shutil.rmtree(st.session_state.temp_dir)

atexit.register(cleanup)
