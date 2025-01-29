import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import shutil

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set. Please check your .env file.")
genai.configure(api_key=api_key)

# Create a directory to store uploaded documents
UPLOAD_DIR = "uploaded_documents"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the context, reply "answer is not available in the context." Do not provide incorrect answers.
    
    Context:\n {context}?\n
    Question:\n {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        # Enable dangerous deserialization since the index file is trusted
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
    except Exception as e:
        st.error(f"Error loading FAISS index: {str(e)}")
        return

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def save_uploaded_files(pdf_docs):
    for pdf in pdf_docs:
        file_path = os.path.join(UPLOAD_DIR, pdf.name)
        with open(file_path, "wb") as f:
            f.write(pdf.getbuffer())
    st.success(f"Saved {len(pdf_docs)} file(s) to {UPLOAD_DIR}")

def list_uploaded_files():
    if os.path.exists(UPLOAD_DIR):
        files = os.listdir(UPLOAD_DIR)
        if files:
            st.write("Previously uploaded documents:")
            for file in files:
                st.write(f"- {file}")
        else:
            st.write("No documents have been uploaded yet.")
    else:
        st.write("No documents have been uploaded yet.")

def main():
    st.set_page_config(page_title="Chat with PDF", layout="wide")
    st.header("Chat with PDF using Gemini💁")

    # Sidebar for file upload and processing
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if pdf_docs and st.button("Submit & Process"):
            with st.spinner("Processing..."):
                try:
                    # Save uploaded files
                    save_uploaded_files(pdf_docs)
                    
                    # Process files
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing complete!")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

        # List previously uploaded documents
        st.subheader("Uploaded Documents")
        list_uploaded_files()

    # Main content area for asking questions
    col1, col2 = st.columns([3, 1])
    with col1:
        user_question = st.text_input("Ask a Question from the PDF Files")
        if user_question:
            user_input(user_question)

    with col2:
        st.write("Tips:")
        st.write("- Upload multiple PDFs to combine their content.")
        st.write("- Ask specific questions for detailed answers.")

if __name__ == "__main__":
    main()