from fastapi import APIRouter
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Import menggunakan path relatif terhadap tempat menjalankan app.py
from rag.loader import load_all_docs
from rag.splitter import split_docs
from rag.embeddings import get_embeddings, create_vector_db
from rag.chain import get_chat_response

load_dotenv()
router = APIRouter()

class ChatRequest(BaseModel):
    message: str

@router.post("/setup-database")
async def setup_database():
    try:
        print("--- Memulai Setup Database ---")
        docs = load_all_docs()
        if not docs:
            return {"error": "File PDF tidak ditemukan di folder docs."}
        
        texts = split_docs(docs)
        # Fungsi ini yang mengirim ke Pinecone
        create_vector_db(texts)
        
        return {"message": f"BERHASIL! {len(texts)} data terkirim ke Pinecone."}
    except Exception as e:
        print(f"Error setup: {str(e)}")
        return {"error": str(e)}

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Panggil fungsi dari chain.py
        answer = get_chat_response(request.message)
        return {"reply": answer}
    except Exception as e:
        return {"reply": f"Maaf, ada kendala teknis: {str(e)}"}