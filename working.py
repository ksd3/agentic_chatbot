import os
import pdfplumber
import pytesseract
from PIL import Image
import numpy as np
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from chromadb import PersistentClient
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Set up Tesseract path for OCR
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Change this as per your environment

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            print(f"Processing page {page_num + 1}/{len(pdf.pages)}")
            page_text = page.extract_text()
            if page_text:
                text += page_text
            else:
                img = page.to_image().original
                ocr_text = pytesseract.image_to_string(Image.fromarray(np.array(img)))
                text += ocr_text
    return text

# Load and process the document
document_path = 'myfile.pdf'
text = extract_text_from_pdf(document_path)
documents = [Document(page_content=text)]

# Initialize OpenAI embeddings
api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# Split document into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Initialize Chroma PersistentClient for vector storage
client = PersistentClient(path="./chroma_vectordb")
client.heartbeat() 
client.heartbeat() 
client.heartbeat() 
client.heartbeat() 
client.heartbeat() 
client.heartbeat() 
client.heartbeat() 
client.heartbeat() 
vectordb = Chroma(client=client).from_documents(docs, embeddings)

# Define the state schema using Pydantic
class StateSchema(BaseModel):
    question: str
    documents: list
    web_search_needed: str = "No"  # Default is "No", can change to "Yes" later

# Define retrieve function
def retrieve(state: StateSchema):
    print("---RETRIEVE DOCUMENTS---")
    query = state.question  # Access the question from the state

    # Retrieve top 5 chunks
    retrieved_docs = vectordb.similarity_search(query, k=5)
    
    # Debugging: Print the retrieved documents to see if anything is returned
    if retrieved_docs:
        print(f"Retrieved Documents: {[doc.page_content[:200] for doc in retrieved_docs]}")  # Only print first 200 chars
    else:
        print("No documents retrieved.")

    return {"documents": retrieved_docs, "question": query}  # Always update documents and question

# Define grade_documents function
def grade_documents(state: StateSchema):
    print("---GRADE DOCUMENTS---")
    docs = state.documents  # Access the documents from the state
    question = state.question  # Access the question from the state
    
    # Grader model
    model = ChatOpenAI(temperature=0, model="gpt-4")
    
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        Provide a binary 'yes' or 'no' to indicate whether the document is relevant.""",
        input_variables=["context", "question"],
    )
    
    graded_docs = []
    
    for doc in docs:
        # Use the invoke method to replace deprecated method
        result = model.invoke(prompt.format(context=doc.page_content, question=question))
        if 'yes' in result.content.lower():
            graded_docs.append(doc)
    
    if not graded_docs:
        print("No relevant documents were graded.")
        return {"documents": [], "web_search_needed": "Yes"}  # Update the state if no relevant documents
    else:
        print(f"Graded documents: {[doc.page_content[:200] for doc in graded_docs]}")  # Debugging: Print relevant docs
    
    return {"documents": graded_docs, "web_search_needed": "No"}  # Update the state with the graded documents

# Define decide_to_generate function
def decide_to_generate(state: StateSchema):
    print("---ASSESS GRADED DOCUMENTS---")
    web_search_needed = state.web_search_needed  # Access the web_search_needed from the state
    if web_search_needed == "Yes":
        return "rewrite_query"
    else:
        return "generate_answer"

# Define rewrite_query function
def rewrite_query(state: StateSchema):
    print("---REWRITE QUERY---")
    question = state.question  # Access the question from the state
    
    msg = [
        HumanMessage(
            content=f"Look at the input question:\n{question}\nFormulate an improved query."
        )
    ]
    
    model = ChatOpenAI(temperature=0, model="gpt-4")
    response = model.invoke(msg)
    
    # Access the content properly from the AIMessage object, not as a dict
    return {"question": response.content}  # Ensure we are updating the question in the state

# Define generate_answer function
def generate_answer(state: StateSchema):
    print("---GENERATE ANSWER---")
    question = state.question  # Access the question from the state
    docs = state.documents  # Access the documents from the state
    
    # Debugging: Ensure that documents and questions are present
    print(f"Question: {question}")
    print(f"Documents for context: {[doc.page_content for doc in docs]}")

    # Load a prompt from LangChain Hub
    prompt = hub.pull("rlm/rag-prompt")
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    rag_chain = prompt | llm | StrOutputParser()

    # Call invoke method properly
    try:
        generation = rag_chain.invoke({"context": docs, "question": question})
        print(f"LLM Generation Result: {generation}")  # Debugging: Print the LLM response
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        return {"generation": None, "documents": docs, "question": question}

    # Ensure generation is added to the state
    if generation and "content" in generation:
        return {"generation": generation["content"], "documents": docs, "question": question}
    else:
        return {"generation": None, "documents": docs, "question": question}

# Initialize the state graph with the schema
agentic_rag = StateGraph(state_schema=StateSchema)

# Add nodes and edges
agentic_rag.add_node("retrieve", retrieve)
agentic_rag.add_node("grade_documents", grade_documents)
agentic_rag.add_node("rewrite_query", rewrite_query)
agentic_rag.add_node("generate_answer", generate_answer)

# Set up the graph flow
agentic_rag.set_entry_point("retrieve")
agentic_rag.add_edge("retrieve", "grade_documents")
agentic_rag.add_conditional_edges(
    "grade_documents", decide_to_generate, 
    {"rewrite_query": "rewrite_query", "generate_answer": "generate_answer"}
)
agentic_rag.add_edge("rewrite_query", "generate_answer")
agentic_rag.add_edge("generate_answer", END)

# Compile the graph
agentic_rag = agentic_rag.compile()

# Test the graph with a query and initial empty documents
query = "What are the key points in the Supreme Court abortion ruling?"
initial_state = {
    "question": query,
    "documents": [],  # Start with an empty document list
    "web_search_needed": "No"  # Initialize with default value
}

# Execute the graph
response = agentic_rag.invoke(initial_state)

# Check if 'generation' exists before accessing
if 'generation' in response and response['generation']:
    print("Generated Answer: ", response["generation"])
else:
    print("No answer generated.")
