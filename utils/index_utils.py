import os
from langchain_community.vectorstores import FAISS
from config.settings import INDEX_DIR

def save_index_bundle(documents, embedder, index_name):
    index_dir = os.path.join(INDEX_DIR, index_name)
    os.makedirs(index_dir, exist_ok=True)
    print(f"💾 [DEBUG] Salvando índice FAISS em: {index_dir}")
    vectorstore = FAISS.from_documents(documents, embedder)
    vectorstore.save_local(index_dir)

def load_index_bundle(index_name, embedder):
    index_dir = os.path.join(INDEX_DIR, index_name)
    print(f"📥 [DEBUG] Carregando índice FAISS de: {index_dir}")
    return FAISS.load_local(index_dir, embedder, allow_dangerous_deserialization=True)

def list_index_names():
    return [name for name in os.listdir(INDEX_DIR) if os.path.isdir(os.path.join(INDEX_DIR, name))]
