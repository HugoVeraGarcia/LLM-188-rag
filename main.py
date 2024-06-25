import streamlit as st
import sys
from PyPDF2 import PdfReader
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

# Ensure sqlite3 uses pysqlite3
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

def generate_response(file, openai_api_key, query):
    try:
        reader = PdfReader(file)
        formatted_document = [page.extract_text() for page in reader.pages]
        
        # Split the document into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.create_documents(formatted_document)
        
        # Create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Store embeddings in vector store
        store = FAISS.from_documents(docs, embeddings)
        
        # Create retrieval chain
        retrieval_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(temperature=0, openai_api_key=openai_api_key),
            chain_type="stuff",
            retriever=store.as_retriever()
        )
        
        # Run chain with query
        return retrieval_chain.run(query)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Streamlit app configuration
st.set_page_config(page_title="Q&A from a long PDF Document")
st.title("Q&A from a long PDF Document")

uploaded_file = st.file_uploader("Upload a .pdf document", type="pdf")
query_text = st.text_input("Enter your question:", placeholder="Write your question here", disabled=not uploaded_file)

with st.form("myform", clear_on_submit=True):
    openai_api_key = st.text_input("OpenAI API Key:", type="password", disabled=not (uploaded_file and query_text))
    submitted = st.form_submit_button("Submit", disabled=not (uploaded_file and query_text))
    
    if submitted and openai_api_key.startswith("sk-"):
        with st.spinner("Wait, please. I am working on it..."):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            if response:
                st.info(response)
            del openai_api_key  # Ensure the API key is removed from memory

if not uploaded_file:
    st.warning("Please upload a PDF file to proceed.")
if not query_text:
    st.warning("Please enter a question to proceed.")
