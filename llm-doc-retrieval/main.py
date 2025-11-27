import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from utils import get_vectorstore

# === ENV SETUP ===
load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')

# === CONFIG ===
PERSIST_DIR = './chroma_db'
PDF_FOLDER = './pdf_folder'
COLLECTION_NAME = 'id_stock_annual_reports'

# === EMBEDDING MODEL ===
BGE_INSTRUCTION = "Represent this sentence for searching relevant passages: "

embeddings = HuggingFaceEmbeddings(
    model_name='BAAI/bge-base-en-v1.5',
    model_kwargs={'device': 'cuda'},
    encode_kwargs={
        # Add instructions as part of BGE best practice
        'prompt': BGE_INSTRUCTION,
        # This is important for BGE's cosine similarity computation
        'normalize_embeddings': True
    }
)

def main():
    vector_store = get_vectorstore(
        persist_dir=PERSIST_DIR,
        embeddings=embeddings,
        collection_name=COLLECTION_NAME
    )

    # === RETRIEVAL ===
    retriever = vector_store.as_retriever(search_kwargs={'k': 20})
    llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', google_api_key=API_KEY)

    system_prompt = '''
    You are a helpful company assistant. Answer the user's question **only** using the provided context.
    Never make up or assume information.
    '''

    prompt = ChatPromptTemplate.from_template(f'''
    {system_prompt}
    Context:
    {{context}}\n
    Question: {{question}}\n
    Answer:
    ''')

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            'prompt': prompt
        }
    )

    # === ASK A QUESTION ===
    while True:
        question = input('\nAsk a question (or type "exit"): ')
        
        if question.lower() in ['exit', 'quit', 'q']:
            print('Exiting program. Goodbye!')
            break
        
        result = qa_chain.invoke(question)

        print(f'\nAnswer: {result['result']}')

        # Show which chunks were used
        for doc in result['source_documents']:
            meta = doc.metadata
            print(f'Source: {meta['source']} (page {meta.get('page', '?')})')

if __name__ == "__main__":
    main()
