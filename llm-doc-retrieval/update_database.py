import os
from utils import get_vectorstore, update_or_add_pdf
from langchain_huggingface import HuggingFaceEmbeddings

# === CONFIG ===
PERSIST_DIR = './chroma_db'
PDF_FOLDER = './pdf_files'
COLLECTION_NAME = 'id_stock_annual_reports'

# === EMBEDDING MODEL ===
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

def main():
    vector_store = get_vectorstore(
        persist_dir=PERSIST_DIR,
        embeddings=embeddings,
        collection_name=COLLECTION_NAME
    )

    # Example: add or update PDFs in folder
    for fname in os.listdir(PDF_FOLDER):
        if not fname.lower().endswith('.pdf'):
            continue
        
        update_or_add_pdf(
            vector_store=vector_store,
            file_path=os.path.join(PDF_FOLDER, fname),
            uploader='steven'
        )

    # # Alternatively, can update a specific file
    # update_or_add_pdf(
    #     vector_store=vector_store,
    #     file_path='file_path',
    #     uploader='steven'
    # )

if __name__ == '__main__':
    main()