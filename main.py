import streamlit as st
import os
import pdfplumber
import pytesseract
from PIL import Image
import numpy as np
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma  # Updated Chroma import
from chromadb import PersistentClient
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from streamlit_chromadb_connection.chromadb_connection import ChromadbConnection
from typing import Optional

# Load environment variables
load_dotenv()

# Set up Tesseract path for OCR
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Adjust this path

# Streamlit app configuration
st.title("LLM-Powered Document Retrieval Chatbot")
st.write("Upload a PDF and ask questions based on the extracted content.")

# File uploader for PDF documents
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Initialize variables for the chatbot
api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

if uploaded_file is not None:
    # Extract text from the uploaded PDF file
    def extract_text_from_pdf(pdf_path):
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                else:
                    img = page.to_image().original
                    ocr_text = pytesseract.image_to_string(Image.fromarray(np.array(img)))
                    text += ocr_text
        return text

    # Save uploaded file and extract text
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    text = extract_text_from_pdf("uploaded.pdf")

    # Create a LangChain document object
    documents = [Document(page_content=text)]
    
    # Split the text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Set up Chroma client for persistent vector storage
    client = PersistentClient(path="./chroma_vectordb")
    vectordb = Chroma(client=client).from_documents(docs, embeddings)

    # Define the state schema for LangChain
    class StateSchema(BaseModel):
        question: str
        documents: list
        web_search_needed: str = "No"
        generation: Optional[str] = None  # Add generation to the state schema

    # Define the chatbot logic (retrieve, grade, generate answer)
    def retrieve(state: StateSchema):
        query = state.question  # Access the question from the state

        # Retrieve top 5 chunks
        retrieved_docs = vectordb.similarity_search(query, k=5)
        return {"documents": retrieved_docs, "question": query}  # Always update documents and question

    def grade_documents(state: StateSchema):
        docs = state.documents
        question = state.question
        model = ChatOpenAI(temperature=0.8, model="gpt-4")  # Slightly higher temperature to allow flexibility

        # Adjusted prompt for leniency
        prompt = """You are assessing whether the following document is relevant to a user question. \n 
                    Here is the document: \n\n {context} \n\n
                    Here is the user question: {question} \n
                    Reply 'yes' if there is any chance this document could be relevant to answering the question. 
                    Only reply 'no' if you are certain the document is not relevant."""

        graded_docs = []
        
        for doc in docs:
            try:
                result = model.invoke(prompt.format(context=doc.page_content, question=question))

                # If the model explicitly says 'no', exclude the document; otherwise, include it
                if 'no' not in result.content.lower():
                    graded_docs.append(doc)
            except Exception as e:
                pass  # Silently ignore errors for now
        
        # If no relevant documents are found, signal that a web search is needed
        return {"documents": graded_docs, "web_search_needed": "No" if graded_docs else "Yes"}

    def generate_answer(state: StateSchema):
        question = state.question  # Access the question from the state
        docs = state.documents  # Access the documents from the state
        
        if not docs:
            return {"generation": "No relevant documents found to answer the question.", "documents": docs, "question": question}

        # Prepare the document content for passing into the model
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = hub.pull("rlm/rag-prompt")
        
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        rag_chain = prompt | llm | StrOutputParser()

        try:
            # Call invoke method properly, ensuring correct input format
            generation = rag_chain.invoke({"context": context, "question": question})
            return {"generation": generation, "documents": docs, "question": question}
            
        except Exception as e:
            return {"generation": None, "documents": docs, "question": question}

    # Define the chatbot logic flow using StateGraph
    agentic_rag = StateGraph(state_schema=StateSchema)
    agentic_rag.add_node("retrieve", retrieve)
    agentic_rag.add_node("grade_documents", grade_documents)
    agentic_rag.add_node("generate_answer", generate_answer)
    agentic_rag.set_entry_point("retrieve")
    agentic_rag.add_edge("retrieve", "grade_documents")
    agentic_rag.add_edge("grade_documents", "generate_answer")
    agentic_rag = agentic_rag.compile()

    # User query input
    user_query = st.text_input("Ask a question about the document:")

    if st.button("Submit Query"):
        if user_query:
            # Initial state setup with user query
            initial_state = {"question": user_query, "documents": [], "web_search_needed": "No"}
            
            # Invoke the agentic graph with the initial state
            response = agentic_rag.invoke(initial_state)
            
            # Display the generated answer if available
            if 'generation' in response and response['generation']:
                st.write(response['generation'])
            else:
                st.write("No relevant answer generated or no documents retrieved.")
