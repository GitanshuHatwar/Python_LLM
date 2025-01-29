import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import shutil
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.vectorstores import FAISS  # This ensures compatibility

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ GOOGLE_API_KEY is missing. Please check your .env file or Streamlit secrets.")
    st.stop()

genai.configure(api_key=api_key)

# Ensure FAISS works correctly
try:
    import faiss
except ModuleNotFoundError:
    st.error("❌ FAISS is not installed. Install it using `pip install faiss-cpu`.")
    st.stop()

# Create directory for uploaded documents
UPLOAD_DIR = "uploaded_documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Function to create a FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    try:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        st.success("✅ FAISS index created successfully!")
    except Exception as e:
        st.error(f"❌ Error while creating FAISS index: {str(e)}")

# Function to load QA chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the context, reply: "The answer is not available in the provided context."
    
    Context:\n {context}?\n
    Question:\n {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Function to process user questions
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
    except Exception as e:
        st.error(f"❌ Error loading FAISS index: {str(e)}")
        return

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("🗨️ Reply:", response["output_text"])

# Function to save uploaded files
def save_uploaded_files(pdf_docs):
    for pdf in pdf_docs:
        file_path = os.path.join(UPLOAD_DIR, pdf.name)
        with open(file_path, "wb") as f:
            f.write(pdf.getbuffer())
    st.success(f"✅ Saved {len(pdf_docs)} file(s) to {UPLOAD_DIR}")

# Function to list previously uploaded documents
def list_uploaded_files():
    files = os.listdir(UPLOAD_DIR)
    if files:
        st.write("📂 Previously uploaded documents:")
        for file in files:
            st.write(f"- {file}")
    else:
        st.write("📁 No documents uploaded yet.")

# Main Streamlit App
def main():
    st.set_page_config(page_title="Chat with PDF", layout="wide")
    st.header("📚 Chat with PDF using Gemini AI 💡")

    # Sidebar for file upload and processing
    with st.sidebar:
        st.title("📌 Menu:")
        pdf_docs = st.file_uploader("📂 Upload PDF Files", accept_multiple_files=True)

        if pdf_docs and st.button("📥 Submit & Process"):
            with st.spinner("⏳ Processing..."):
                try:
                    save_uploaded_files(pdf_docs)
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ Processing complete! You can now ask questions.")
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")

        # Show uploaded documents
        st.subheader("📄 Uploaded Documents")
        list_uploaded_files()

    # Main content for asking questions
    col1, col2 = st.columns([3, 1])
    with col1:
        user_question = st.text_input("💬 Ask a Question")
        if user_question:
            user_input(user_question)

    with col2:
        st.write("💡 Tips:")
        st.write("- Upload multiple PDFs for combined content analysis.")
        st.write("- Ask clear, specific questions for better answers.")

if __name__ == "__main__":
    main()
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import shutil
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ GOOGLE_API_KEY is missing. Please check your .env file or Streamlit secrets.")
    st.stop()

genai.configure(api_key=api_key)

# Ensure FAISS works correctly
try:
    import faiss
except ModuleNotFoundError:
    st.error("❌ FAISS is not installed. Install it using `pip install faiss-cpu`.")
    st.stop()

# Create directory for uploaded documents
UPLOAD_DIR = "uploaded_documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Function to create a FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    try:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        st.success("✅ FAISS index created successfully!")
    except Exception as e:
        st.error(f"❌ Error while creating FAISS index: {str(e)}")

# Function to load QA chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the context, reply: "The answer is not available in the provided context."
    
    Context:\n {context}?\n
    Question:\n {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Function to process user questions
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
    except Exception as e:
        st.error(f"❌ Error loading FAISS index: {str(e)}")
        return

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("🗨️ Reply:", response["output_text"])

# Function to save uploaded files
def save_uploaded_files(pdf_docs):
    for pdf in pdf_docs:
        file_path = os.path.join(UPLOAD_DIR, pdf.name)
        with open(file_path, "wb") as f:
            f.write(pdf.getbuffer())
    st.success(f"✅ Saved {len(pdf_docs)} file(s) to {UPLOAD_DIR}")

# Function to list previously uploaded documents
def list_uploaded_files():
    files = os.listdir(UPLOAD_DIR)
    if files:
        st.write("📂 Previously uploaded documents:")
        for file in files:
            st.write(f"- {file}")
    else:
        st.write("📁 No documents uploaded yet.")

# Main Streamlit App
def main():
    st.set_page_config(page_title="Chat with PDF", layout="wide")
    st.header("📚 Chat with PDF using Gemini AI 💡")

    # Sidebar for file upload and processing
    with st.sidebar:
        st.title("📌 Menu:")
        pdf_docs = st.file_uploader("📂 Upload PDF Files", accept_multiple_files=True)

        if pdf_docs and st.button("📥 Submit & Process"):
            with st.spinner("⏳ Processing..."):
                try:
                    save_uploaded_files(pdf_docs)
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ Processing complete! You can now ask questions.")
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")

        # Show uploaded documents
        st.subheader("📄 Uploaded Documents")
        list_uploaded_files()

    # Main content for asking questions
    col1, col2 = st.columns([3, 1])
    with col1:
        user_question = st.text_input("💬 Ask a Question")
        if user_question:
            user_input(user_question)

    with col2:
        st.write("💡 Tips:")
        st.write("- Upload multiple PDFs for combined content analysis.")
        st.write("- Ask clear, specific questions for better answers.")

if __name__ == "__main__":
    main()
