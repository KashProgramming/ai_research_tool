import os
import streamlit as st
import pickle
import time
import langchain
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

st.title("Research Tool")

st.sidebar.title("Article URLs")

urls = []
for i in range(4):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placefolder = st.empty()
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=st.secrets["DEEPSEEK_API_KEY"],
    model_name="deepseek/deepseek-chat:free"
)

if process_url_clicked:
    # loading the data
    loader = UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Data Loading started...")
    data = loader.load()
    # splitting the data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n","."," "],
        chunk_size=1000
    )
    main_placefolder.text("Text splitting started...")
    docs = text_splitter.split_documents(data)
    # creating embeddings and saving them to FAISS index
    embeddings = HuggingFaceEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placefolder.text("Vector embedding started building...")
    time.sleep(2)

    with open(file_path,"wb") as f:
        pickle.dump(vectorstore_openai,f)

query = main_placefolder.text_input("Questions: ")
if query:
    if os.path.exists(file_path):
        with open(file_path,"rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs = True)
            st.header("Answer:")
            st.write(result["answer"])
            sources = result.get("sources","")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
