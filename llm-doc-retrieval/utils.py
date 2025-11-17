import os
from datetime import datetime
import pytz

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# === UTILITY: Extract product name from filename ===
def product_name_from_filename(filename: str) -> str:
    name = os.path.splitext(os.path.basename(filename))[0]
    return name

# === PDF INGESTION WITH METADATA ===
def ingest_pdf_to_docs(file_path: str, uploader: str='automated'):
    '''Load a PDF, split into chunks, enrich with metadata.'''
    loader = PyPDFLoader(file_path)
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(raw_docs)

    source = os.path.basename(file_path)
    product = product_name_from_filename(source)

    tz_jkt = pytz.timezone('Asia/Jakarta')
    ingested_at_jkt = datetime.now(tz=tz_jkt).replace(tzinfo=None).isoformat()

    for i, chunk in enumerate(chunks):
        meta = dict(chunk.metadata or {})
        page = meta.get('page', None)

        meta.upadate({
            'source': source,
            'product': product,
            'page': page,
            'chunk_id': i,
            'type': 'pdf',
            'uploaded_by': uploader,
            'uploaded_at': ingested_at_jkt
        })

        chunk.metadata = meta

        # deterministic ID for updates / de-duplication
        chunk.metadata['unique_id'] = f'{product}::page{page if page is not None else 'na'}::chunk{i}'
    
    return chunks


# === CREATE OR LOAD VECTOR STORE ===
def get_vectorstore(persist_dir, embeddings, collection_name) -> Chroma:
    if os.path.exists(persist_dir):
        print('Loading existing vector store from disk...')
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=collection_name
        )
    
    else:
        print('Creating new vector store...')
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=collection_name
        )

# === UPDATE OR ADD A PDF ===
def update_or_add_pdf(vector_store: Chroma, file_path: str, uploader: str):
    '''Replace old embeddings for this PDF (if exist) and add the new version'''
    product = product_name_from_filename(file_path)
    print(f'Processing: {product}')

    chunks = ingest_pdf_to_docs(file_path, uploader)
    texts = [d.page_content for d in chunks]
    metadatas = [d.metadata for d in chunks]
    ids = [m['unique_id'] for m in metadatas]

    # delete old ones if exist
    try:
        vector_store.delete(ids=ids)
        print(f'Deleted old entries for {product}.')
    except Exception:
        pass # if not exist, no issue

    # add new embeddings
    vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    print(f'Added {len(ids)} new chunks for {product}.')