from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_docs(documents):
    # chunk_size: jumlah karakter per potongan
    # chunk_overlap: potongan yang bertumpuk agar konteks tidak terputus
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Dokumen dipecah menjadi {len(chunks)} potongan kecil.")
    return chunks