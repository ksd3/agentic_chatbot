import streamlit as st
import os
import pdfplumber
import pytesseract
from PIL import Image
import numpy as np
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from chromadb import PersistentClient
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
from langgraph.graph import StateGraph
from typing import Optional

# Load environment variables from a .env file (for example, OpenAI API keys)
load_dotenv()

# Set up the Tesseract path for OCR (Optical Character Recognition)
# This path may need to be adjusted depending on the environment (Windows, Linux, MacOS)
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Update this path based on your setup

# Streamlit app configuration
st.title("LLM-Powered Document Retrieval Chatbot")  # Set the title of the web app
st.write("Upload a PDF and ask questions based on the extracted content.")  # Brief description

# File uploader for PDF documents, limiting upload types to only PDF files
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Initialize API key for OpenAI and embeddings model
api_key = os.getenv("OPENAI_API_KEY")  # Retrieve OpenAI API key from environment
embeddings = OpenAIEmbeddings(openai_api_key=api_key)  # Initialize embeddings model using OpenAI

# Check if 'chat_history' exists in session state; if not, initialize it as an empty list
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []  # This will store the conversation history (user and bot messages)

# If a PDF file is uploaded, we begin processing the file
if uploaded_file is not None:

    # Function to extract text from an uploaded PDF
    def extract_text_from_pdf(pdf_path):
        """
        Extracts text from a PDF file using both text extraction and OCR (Optical Character Recognition) for images.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            str: The extracted text from the PDF file.
        """
        text = ""  # Initialize an empty string to store extracted text
        with pdfplumber.open(pdf_path) as pdf:  # Open the PDF file using pdfplumber
            # Iterate through each page in the PDF
            for page_num, page in enumerate(pdf.pages):
                # Try to extract text from the page
                page_text = page.extract_text()
                if page_text:  # If text is found, append it to 'text'
                    text += page_text
                else:  # If no text is found (e.g., scanned images), perform OCR
                    img = page.to_image().original  # Convert the page to an image
                    ocr_text = pytesseract.image_to_string(Image.fromarray(np.array(img)))  # Perform OCR
                    text += ocr_text  # Append the OCR text to 'text'
        return text  # Return the final extracted text from all pages

    # Save the uploaded PDF file locally and extract text from it
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())  # Write the uploaded PDF to a file
    text = extract_text_from_pdf("uploaded.pdf")  # Extract text from the saved PDF file

    # Create a LangChain document object, which contains the extracted text
    documents = [Document(page_content=text)]
    
    # Split the extracted text into manageable chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)  # This splits the document into smaller chunks (overlapping for context)

    # Initialize a Chroma client for persistent vector storage (to store embeddings)
    client = PersistentClient(path="./chroma_vectordb")  # Path to persistent storage for vector DB
    vectordb = Chroma(client=client).from_documents(docs, embeddings)  # Store document embeddings in Chroma DB

    # Define the schema for the chatbot's state (keeps track of the user query, retrieved docs, etc.)
    class StateSchema(BaseModel):
        question: str  # The user's question or query
        documents: list  # List of relevant documents or chunks
        web_search_needed: str = "No"  # Flag to indicate if web search is needed (default is No)
        generation: Optional[str] = None  # The generated answer from the model

    # Function to retrieve relevant document chunks based on the user's query
    def retrieve(state: StateSchema):
        """
        Retrieves relevant document chunks based on the user's query by performing similarity search.

        Args:
            state (StateSchema): The current state of the chatbot, including the user query.

        Returns:
            dict: A dictionary containing the retrieved documents and the user query.
        """
        query = state.question  # Extract the user query from the state

        # Perform a similarity search in the vector database to find the top 5 most relevant chunks
        retrieved_docs = vectordb.similarity_search(query, k=5)
        return {"documents": retrieved_docs, "question": query}  # Return the retrieved documents and query

    # Function to grade (assess) the relevance of retrieved documents using the LLM
    def grade_documents(state: StateSchema):
        """
        Grades the relevance of the retrieved documents to the user's question using the LLM (Language Learning Model).

        Args:
            state (StateSchema): The current state of the chatbot, including the user query and documents.

        Returns:
            dict: A dictionary with graded documents and a flag indicating if a web search is needed.
        """
        docs = state.documents  # Extract the list of retrieved documents
        question = state.question  # Extract the user query
        model = ChatOpenAI(temperature=0.8, model="gpt-4")  # Initialize the OpenAI model (GPT-4 with temperature for variability)

        # Define the prompt for grading document relevance
        prompt = """You are assessing whether the following document is relevant to a user question. \n 
                    Here is the document: \n\n {context} \n\n
                    Here is the user question: {question} \n
                    Reply 'yes' if there is any chance this document could be relevant to answering the question. 
                    Only reply 'no' if you are certain the document is not relevant."""

        graded_docs = []  # List to store the graded (relevant) documents
        
        # Iterate over each document and use the model to assess relevance
        for doc in docs:
            try:
                # Generate a model response for each document's context and question
                result = model.invoke(prompt.format(context=doc.page_content, question=question))

                # If the model's response doesn't contain 'no', consider the document relevant
                if 'no' not in result.content.lower():
                    graded_docs.append(doc)  # Append relevant documents to the graded list
            except Exception as e:
                pass  # Handle errors silently
        
        # Return graded documents and indicate whether a web search is needed (if no relevant documents are found)
        return {"documents": graded_docs, "web_search_needed": "No" if graded_docs else "Yes"}

    # Function to generate an answer based on the user's query and the relevant documents
    def generate_answer(state: StateSchema):
        """
        Generates an answer to the user's query based on the relevant documents using the LLM.

        Args:
            state (StateSchema): The current state of the chatbot, including the user query and documents.

        Returns:
            dict: A dictionary containing the generated answer, documents, and the user query.
        """
        question = state.question  # Extract the user query
        docs = state.documents  # Extract the list of relevant documents
        
        # If no relevant documents are available, return a default response
        if not docs:
            return {"generation": "No relevant documents found to answer the question.", "documents": docs, "question": question}

        # Prepare the context (document content) by concatenating the page content of all relevant documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Pull the prompt from LangChain Hub (this can be customized based on task)
        prompt = hub.pull("rlm/rag-prompt")
        
        # Initialize the OpenAI model (ChatGPT-3.5 Turbo with low temperature for consistency)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        rag_chain = prompt | llm | StrOutputParser()  # Chain the prompt with the model and output parser

        try:
            # Generate an answer by passing the context and question to the LLM
            generation = rag_chain.invoke({"context": context, "question": question})
            return {"generation": generation, "documents": docs, "question": question}
        except Exception as e:
            # Return None for generation in case of an error
            return {"generation": None, "documents": docs, "question": question}

    # Function to check for inappropriate content in the user's query (e.g., violence, sexual content)
    def check_inappropriate_content(query):
        """
        Checks if the user query contains inappropriate content such as violence, sex, drugs, hate speech, etc.

        Args:
            query (str): The user query.

        Returns:
            bool: True if inappropriate content is detected, False otherwise.
        """
        # List of inappropriate keywords to check for in the query
        inappropriate_keywords = ['violence', 'sex', 'drugs', 'hate', 'abuse']
        for word in inappropriate_keywords:
            if word in query.lower():  # Check if any inappropriate keyword exists in the query (case-insensitive)
                return True
        return False  # Return False if no inappropriate content is found

    # Function to check if the user's query is irrelevant (e.g., asking about coding in a PDF-related chatbot)
    def check_irrelevant_question(query):
        """
        Checks if the user query is irrelevant to the chatbot's purpose (e.g., asking about coding when the chatbot is focused on PDFs).

        Args:
            query (str): The user query.

        Returns:
            bool: True if the query is irrelevant, False otherwise.
        """
        # List of irrelevant keywords (e.g., programming-related keywords) to check for in the query
        irrelevant_keywords = ['code', 'python', 'java', 'programming', 'software']
        for word in irrelevant_keywords:
            if word in query.lower():  # Check if the query contains any irrelevant keyword (case-insensitive)
                return True
        return False  # Return False if the query is relevant

    # Define the chatbot logic flow using LangChain's StateGraph
    agentic_rag = StateGraph(state_schema=StateSchema)  # Initialize the graph with the defined state schema
    agentic_rag.add_node("retrieve", retrieve)  # Add the 'retrieve' function as the first node in the graph
    agentic_rag.add_node("grade_documents", grade_documents)  # Add the 'grade_documents' function as the second node
    agentic_rag.add_node("generate_answer", generate_answer)  # Add the 'generate_answer' function as the third node
    agentic_rag.set_entry_point("retrieve")  # Set the 'retrieve' function as the starting point of the graph
    agentic_rag.add_edge("retrieve", "grade_documents")  # Define a transition from 'retrieve' to 'grade_documents'
    agentic_rag.add_edge("grade_documents", "generate_answer")  # Define a transition from 'grade_documents' to 'generate_answer'
    agentic_rag = agentic_rag.compile()  # Compile the graph for execution

    # User input for the query/question
    user_query = st.text_input("Ask a question about the document:")

    # When the user clicks the 'Submit Query' button
    if st.button("Submit Query"):
        if user_query:  # Check if the user has entered a query
            # First, check if the query contains inappropriate content
            if check_inappropriate_content(user_query):
                st.write("Sorry, your question contains inappropriate content. Please ask a different question.")
            # Next, check if the query is irrelevant to the chatbot's purpose
            elif check_irrelevant_question(user_query):
                st.write("This chatbot is meant to answer questions related to the uploaded PDF. Please ask a relevant question.")
            else:
                # Initialize the chatbot's state with the user's query
                initial_state = {"question": user_query, "documents": [], "web_search_needed": "No"}
                
                # Invoke the graph (starting from 'retrieve') with the initial state
                response = agentic_rag.invoke(initial_state)

                # If a response is generated, append the user's query and the bot's response to the chat history
                if 'generation' in response and response['generation']:
                    st.session_state.chat_history.append({"user": user_query, "bot": response['generation']})

    # Display the chat history (all previous user and bot interactions)
    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            st.write(f"**User:** {chat['user']}")  # Display the user's query
            st.write(f"**Bot:** {chat['bot']}")  # Display the bot's response
            st.write("---")  # Add a divider between each chat exchange
