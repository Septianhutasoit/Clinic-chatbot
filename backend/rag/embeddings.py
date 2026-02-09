import os
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Fungsi untuk mengambil model embedding (digunakan saat Chat & Ingest)
def get_embeddings():
    return CohereEmbeddings(model="embed-multilingual-v3.0")

# Fungsi untuk upload data ke Pinecone
def create_vector_db(chunks):
    index_name = os.getenv("PINECONE_INDEX")
    if not index_name:
        print("❌ Error: PINECONE_INDEX belum diatur di .env")
        return None

    try:
        embeddings = get_embeddings()
        vector_db = PineconeVectorStore.from_documents(
            chunks, 
            embeddings, 
            index_name=index_name
        )
        print(f"🚀 BERHASIL! Data sudah tersimpan di Pinecone.")
        return vector_db
    except Exception as e:
        print(f"❌ Terjadi kesalahan: {e}")
        return None