import os
from langchain_cohere import ChatCohere
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    # Pastikan model yang digunakan adalah 'command-r' 
    # karena model 'command' sudah dihapus per Sept 2025
    llm = ChatCohere(
        model="command-r-plus", 
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        temperature=0.3
    )
    return llm