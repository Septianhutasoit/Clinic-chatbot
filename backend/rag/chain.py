import os
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --- 1. INISIALISASI GLOBAL (HANYA JALAN 1X) ---
# Menggunakan temperature 0 agar AI sangat kaku/patuh pada data (tidak kreatif)
embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
index_name = os.getenv("PINECONE_INDEX")
vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
llm = ChatCohere(model="command-r-08-2024", temperature=0) # Temp 0 = Paling Akurat

def get_chat_response(user_query):
    try:
        # --- 2. PROMPT ENGINEERING TINGKAT LANJUT ---
        # Kita tambahkan instruksi "Haram" untuk menjawab di luar teks
        prompt = ChatPromptTemplate.from_template("""
        Anda adalah Asisten AI Resmi dari Klinik Gigi Senyum Sehat.
        
        SISTEM KERJA:
        1. Analisa KONTEKS yang diberikan. Informasi jadwal, nama dokter, dan layanan ada di sana.
        2. Jawablah pertanyaan pasien HANYA menggunakan informasi dari KONTEKS.
        3. Jika informasi TIDAK TERCANTUM secara eksplisit di KONTEKS, katakan: 
           "Mohon maaf, saya tidak menemukan informasi tersebut di data jadwal kami. Silakan hubungi Admin di 0812-3456-7890."
        4. JANGAN PERNAH berasumsi atau mengarang nama dokter atau jam praktek.
        5. Abaikan pengetahuan umum Anda tentang jadwal klinik lain.

        KONTEKS DATA KLINIK:
        {context}
        
        PERTANYAAN PASIEN: {input}
        
        JAWABAN ANDA:""")

        # --- 3. RETRIEVER DENGAN FILTER ---
        # k: 5 mengambil 5 potongan teks paling relevan
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        
        # Eksekusi pencarian dan jawaban
        response = retrieval_chain.invoke({"input": user_query})
        
        # --- 4. DEBUG LOGGING (Untuk Memantau Akurasi di Terminal) ---
        print(f"\n[USER]: {user_query}")
        print("--- ISI PDF YANG DIAMBIL PINECONE ---")
        if not response["context"]:
            print("(!) Peringatan: Pinecone tidak menemukan data yang cocok!")
        else:
            for i, doc in enumerate(response["context"]):
                print(f"{i+1}. {doc.page_content[:200]}...") # Print 200 karakter pertama
        print("--------------------------------------\n")

        return response["answer"]
        
    except Exception as e:
        print(f"--- ERROR: {str(e)} ---")
        return "Maaf, sedang ada gangguan pada sistem chat. Silakan coba lagi nanti."