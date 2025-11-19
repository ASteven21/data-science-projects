import chromadb
import pandas as pd

client = chromadb.PersistentClient('./chroma_db')
collection = client.get_collection('id_stock_annual_reports')
data = collection.get(include=['documents', 'metadatas'])

df = pd.DataFrame({
    'id': data['ids'],
    'document': data['documents'],
    'metadata': data['metadatas']
})

print(df.head())