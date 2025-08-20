import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import os
import tempfile
import shutil
import torch
from dotenv import load_dotenv

load_dotenv()
# os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
groq_api=st.secrets["GROQ_API_KEY"]

torch.classes.__path__=[]

# Page config
st.set_page_config(page_title="Research Assistant")

# Title
st.title("Research Assistant")
st.write("Enter up to 5 article URLs and ask questions about them!")

# Function to extract text from URL
def extract_text_from_url(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and clean it
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
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

# Process URLs button
if st.button("Process") and urls:
    with st.spinner("Processing articles..."):
        documents = []
        
        for url in urls:
            text = extract_text_from_url(url)
            if text:
                # Create document with metadata
                doc = Document(page_content=text, metadata={"source": url})
                documents.append(doc)
            else:
                st.error(f"❌ Failed to process: {url}")
        
        if documents:
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            
            # Create embeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            # Create vector store in temporary directory
            temp_dir = tempfile.mkdtemp()
            chroma_settings=chromadb.config.Settings(
                anonymized_telemetry=False,
                chroma_db_impl="duckdb+parquet"
            )
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=temp_dir,
                client_settings=chroma_settings
            )
            
            # Store in session state
            st.session_state.vectorstore = vectorstore
            st.session_state.temp_dir = temp_dir
            
            st.success(f"Created knowledge base from {len(documents)} articles!")

# Q&A section
if 'vectorstore' in st.session_state:
    st.header("Ask Questions")
    
    question = st.text_input("Enter your question:")
    
    if st.button("Get Answer") and question:
        with st.spinner("Searching for answer..."):
            # Initialize ChatGroq
            llm = ChatGroq(model="llama-3.3-70b-versatile")
            
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )
            
            # Get answer
            result = qa_chain.invoke({"query": question})
            
            # Display answer
            st.subheader("Answer:")
            st.write(result['result'])
            
            # Display sources
            st.subheader("Sources:")
            sources = set()
            for doc in result['source_documents']:
                sources.add(doc.metadata['source'])
            
            for i, source in enumerate(sources, 1):
                st.write(f"{i}. {source}")

# Cleanup function
def cleanup():
    if 'temp_dir' in st.session_state and os.path.exists(st.session_state.temp_dir):
        shutil.rmtree(st.session_state.temp_dir)

# Register cleanup
import atexit
atexit.register(cleanup)
