import os
from dotenv import load_dotenv
from rag.loader import load_all_docs
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings # Pakai Cohere
from pinecone import Pinecone

# Ambil lokasi file .env
current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, ".env")
load_dotenv(dotenv_path)

def push_ke_pinecone():
    # Ambil variabel dari .env kamu
    cohere_key = os.getenv("COHERE_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX") # Sesuai .env kamu

    # Validasi key
    if not cohere_key or not pinecone_key or not index_name:
        print(f"❌ Error: Pastikan COHERE_API_KEY, PINECONE_API_KEY, dan PINECONE_INDEX sudah ada di .env")
        return

    try:
        # 1. Baca semua dokumen (PDF & TXT)
        print("--- Memulai Loading Dokumen ---")
        documents = load_all_docs()
        if not documents:
            print("❌ Tidak ada dokumen untuk diupload.")
            return
        
        # 2. Potong teks (Chunking)
        print("Memotong dokumen menjadi chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        
        # 3. Inisialisasi Embeddings Cohere
        # Gunakan model 'embed-english-v3.0' atau 'embed-multilingual-v3.0' (lebih bagus untuk Indonesia)
        embeddings = CohereEmbeddings(
            cohere_api_key=cohere_key, 
            model="embed-multilingual-v3.0" 
        )
        
        # 4. Inisialisasi Pinecone
        pc = Pinecone(api_key=pinecone_key)
        index = pc.Index(index_name)
        
        # 5. Bersihkan data lama agar tidak duplikat
        print(f"Membersihkan index: {index_name}...")
        index.delete(delete_all=True)
        
        # 6. Upload data baru
        print(f"Mengupload {len(docs)} potongan data ke Pinecone...")
        PineconeVectorStore.from_documents(
            docs, 
            embeddings, 
            index_name=index_name,
            pinecone_api_key=pinecone_key
        )
        print("\n✅ BERHASIL! Sekarang chatbot sudah bisa menjawab pertanyaan dari PDF.")
        
    except Exception as e:
        print(f"❌ Terjadi kesalahan: {e}")

if __name__ == "__main__":
    push_ke_pinecone()