import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def load_all_docs():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "../../docs/")
    
    documents = []
    print(f"--- Memulai Proses Loading Dokumen ---")
    
    if not os.path.exists(path):
        print(f"Error: Folder {path} tidak ditemukan!")
        return []

    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        try:
            if file.endswith(".pdf"):
                print(f"Membaca PDF: {file}...")
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith(".txt"):
                print(f"Membaca TXT: {file}...")
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
        except Exception as e:
            print(f"⚠️ Gagal membaca {file}. Pastikan file bukan hasil 'rename' manual. Error: {e}")

    print(f"--- Selesai! Berhasil memuat {len(documents)} halaman/dokumen ---\n")
    return documents

if __name__ == "__main__":
    load_all_docs()