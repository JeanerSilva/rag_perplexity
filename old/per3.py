import streamlit as st
import pandas as pd
import os
import pickle
from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from sentence_transformers import CrossEncoder
from sklearn.preprocessing import normalize
import numpy as np
import faiss
from datetime import datetime
import glob
import csv

# Diretórios
DATA_FOLDER = "fonte_de_dados/dados_abertos"
INDEX_DIR = "faiss_indices"
EMBEDDING_CACHE_DIR = "embedding_cache"
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)

# Funções auxiliares

def format_doc_e5(text):
    return "passage: " + text

def normalize_vectors(vectors):
    if vectors.size == 0:
        raise ValueError("Vetor de embeddings está vazio.")
    if len(vectors.shape) != 2:
        raise ValueError(f"Vetor de embeddings deve ser 2D, mas tem shape {vectors.shape}")
    return normalize(vectors, axis=1)

def chunk_text(text, chunk_size=512):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def detect_delimiter(file_path):
    with open(file_path, 'r', encoding='latin1') as f:
        sample = f.read(2048)
        sniffer = csv.Sniffer()
        return sniffer.sniff(sample).delimiter

def read_csvs_from_folder(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    texts = []

    for file in csv_files:
        try:
            delimiter = detect_delimiter(file)
            df = pd.read_csv(file, encoding="latin1", sep=delimiter, engine='python', on_bad_lines='skip')
        except Exception as e:
            print(f"Erro ao ler {file}: {e}")
            continue

        for _, row in df.iterrows():
            row_text = "; ".join([f"{col}: {str(row[col])}" for col in df.columns if pd.notnull(row[col])])
            texts.append(row_text)

    return texts

def cache_embeddings(docs, embedder, cache_path):
    if os.path.exists(cache_path):
        print(f"Carregando embeddings do cache: {cache_path}")
        with open(cache_path, "rb") as f:
            embeddings = pickle.load(f)
        embeddings = np.array(embeddings)
        if embeddings.size == 0 or len(embeddings.shape) != 2:
            raise ValueError("Embeddings no cache estão vazios ou com formato inválido.")
        return embeddings
    else:
        if len(docs) == 0:
            raise ValueError("Lista de documentos está vazia. Nada para embutir.")
        print(f"Gerando embeddings para {len(docs)} documentos...")
        embeddings = embedder.embed_documents(docs)
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("Nenhum embedding foi retornado pelo embedder.")
        embeddings = np.array(embeddings)
        if len(embeddings.shape) != 2:
            raise ValueError(f"Embeddings devem ser 2D, mas receberam shape {embeddings.shape}")
        embeddings = normalize_vectors(embeddings)
        with open(cache_path, "wb") as f:
            pickle.dump(embeddings, f)
        return embeddings

def create_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # Produto interno para similaridade de cosseno (com vetores normalizados)
    index.add(embeddings)
    return index

def save_faiss_index(index, path):
    faiss.write_index(index, path)

def rerank_docs(query, docs, model_name, top_k=5):
    cross_encoder = CrossEncoder(model_name)
    pairs = [[query, doc] for doc in docs]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, docs), reverse=True)
    return [doc for _, doc in ranked[:top_k]]

# Streamlit UI

st.title("Indexação automática dos CSVs da pasta fonte_de_dados/dados_abertos")

embedding_model = st.selectbox("Modelo de Embedding", ["multilingual_e5_large", "bge-base-pt"])
chunk_size = st.slider("Tamanho do chunk (palavras)", 128, 1024, 512, step=64)
index_version = st.text_input("Versão do índice (opcional)", value=datetime.now().strftime("%Y%m%d_%H%M%S"))

if st.button("Indexar todos os CSVs da pasta"):
    with st.spinner("Lendo arquivos CSV e criando indexação..."):
        documents = read_csvs_from_folder(DATA_FOLDER)

        if len(documents) == 0:
            st.error("Nenhum documento encontrado na pasta especificada.")
        else:
            if chunk_size and chunk_size < 512:
                chunked_docs = []
                for doc in documents:
                    chunked_docs.extend(chunk_text(doc, chunk_size))
                documents = chunked_docs

            try:
                if embedding_model == "multilingual_e5_large":
                    # Use the correct model name if it's available in Ollama
                    # embedder = OllamaEmbeddings(model="intfloat/multilingual-e5-large") # Corrected model name
                    # If the model is not available in Ollama, use a local HuggingFace model
                    embedder = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large") # Use HuggingFace model
                else:
                    embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-base-pt")

                cache_path = os.path.join(EMBEDDING_CACHE_DIR, f"{embedding_model}_{index_version}.pkl")
                embeddings = cache_embeddings(documents, embedder, cache_path)

                faiss_index = create_faiss_index(embeddings)
                index_path = os.path.join(INDEX_DIR, f"{embedding_model}_{index_version}.index")
                save_faiss_index(faiss_index, index_path)

                docs_path = os.path.join(INDEX_DIR, f"{embedding_model}_{index_version}_docs.pkl")
                with open(docs_path, "wb") as f:
                    pickle.dump(documents, f)

                st.success(f"Indexação concluída! Índice salvo em: {index_path}")
                st.write(f"Total de documentos indexados: {len(documents)}")
            except Exception as e:
                st.error(f"Erro durante a indexação: {e}")
