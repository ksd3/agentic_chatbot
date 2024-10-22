Implementation of an Agentic RAG chatbot using the OpenAI API in LangGraph/Langchain/ChromaDB, with Streamlit for a frontend.

How to install:

1. Load a custom Python3.10+ venv with ``python -m venv .``
2. Create a .env file containing ``OPENAI_API_KEY='your_key_in_these_single_quotes'``
3. Install the requirements with ``pip install -r requirements.txt``
4. Run ``streamlit run main.py``
5. Load the generated link in a browser

Features:
1. Agentic RAG with LangGraph (just a single agent for proof-of-concept)
2. Persistent chat capabilities, with the ability to remember previous chats
3. Customizable censorship to ban keywords that relate to inappropriate content
4. UI prototype in Streamlit
5. Ability to read PDFs with images with OCR
6. Builds semantic understanding/memory of PDF

Example Output:
![image](https://github.com/user-attachments/assets/f4467841-2e21-4ba0-a660-913a72c1fd2b)

Known Issues:
1. Langchain/Langgraph inheritance errors/warnings
Solution: Ask on the langchain forums. Notorious for never being up to date and breaking with every update.

2. Streamlit doesn't return the output after one click
Solution: Varies from system to system. Keep trying!

