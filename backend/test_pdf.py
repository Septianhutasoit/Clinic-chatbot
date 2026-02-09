# backend/test_pdf.py
from langchain_community.document_loaders import PyPDFLoader
import os

# Sesuaikan path ke file PDF kamu
path = os.path.join(os.path.dirname(__file__), "../docs/jadwal_dokter.pdf")

try:
    loader = PyPDFLoader(path)
    pages = loader.load()
    print(f"Total Halaman: {len(pages)}")
    for i, page in enumerate(pages):
        print(f"\n--- Halaman {i+1} ---\n")
        print(page.page_content[:500]) # Print 500 karakter pertama
except Exception as e:
    print(f"Error: {e}")