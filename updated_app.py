__import__('pysqlite3')
import sys
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

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Load environment variables
load_dotenv()

# API Key Handling
groq_api = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
if not groq_api:
    st.error("GROQ API Key not found! Set it in Streamlit Secrets or .env file.")
    st.stop()

# Page config
st.set_page_config(page_title="Research Assistant")

# Title
st.title("Research Assistant")
st.write("Enter up to 5 article URLs and ask questions about them!")

# Function to extract text from URL
def extract_text_from_url(url):
    try:
        if not url.startswith("http"):
            return None

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Remove scripts and styles
        for script in soup(["script", "style"]):
            script.decompose()

        # Extract text mainly from paragraphs
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        text = " ".join(paragraphs)

        return text if text else None
    except Exception as e:
        st.error(f"Error extracting text from {url}: {str(e)}")
        return None

# URL input section
st.header("Enter Articles")
urls = []
for i in range(5):
    url = st.text_input(f"URL {i+1}:", key=f"url_{i}")
    if url:
        urls.append(url)

# Process URLs
if st.button("Process") and urls:
    with st.spinner("Processing articles..."):
        documents = []
        for url in urls:
            text = extract_text_from_url(url)
            if text:
                doc = Document(page_content=text, metadata={"source": url})
                documents.append(doc)
            else:
                st.error(f"Failed to process: {url}")

        if documents:
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)

            # Create embeddings (cached)
            if "embeddings" not in st.session_state:
                st.session_state.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    cache_folder="models"
                )

            # Create vector store in temporary directory
            temp_dir = tempfile.mkdtemp()
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=st.session_state.embeddings,
                persist_directory=temp_dir
            )

            # Store in session state
            st.session_state.vectorstore = vectorstore
            st.session_state.temp_dir = temp_dir

            # Initialize LLM and QA Chain
            st.session_state.llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api)
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=st.session_state.llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )

            st.success(f"Created knowledge base from {len(documents)} articles!")

# Q&A Section
if "qa_chain" in st.session_state:
    st.header("Ask Questions")
    question = st.text_input("Enter your question:")

    if question and st.button("Get Answer"):
        with st.spinner("Searching for answer..."):
            result = st.session_state.qa_chain.invoke({"query": question})

            st.subheader("Answer:")
            st.write(result["result"])

            st.subheader("Sources:")
            sources = {doc.metadata["source"] for doc in result["source_documents"]}
            for i, source in enumerate(sources, 1):
                st.write(f"{i}. {source}")

# Cleanup function for temporary directory
def cleanup():
    if "temp_dir" in st.session_state and os.path.exists(st.session_state.temp_dir):
        shutil.rmtree(st.session_state.temp_dir)

atexit.register(cleanup)
