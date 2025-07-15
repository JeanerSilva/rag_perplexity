import streamlit as st
import pandas as pd
import os
import pickle
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from sklearn.preprocessing import normalize
import numpy as np
from datetime import datetime

# --- Configuração inicial ---
st.set_page_config(page_title="Analisador PLURIANUAL", layout="wide")
INDEX_DIR = "faiss_indices"
EMBEDDING_CACHE_DIR = "embedding_cache"
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)

# --- Funções auxiliares ---
def format_query_e5(text):
    return "query: " + text

def format_doc_e5(text):
    return "passage: " + text

def normalize_vectors(vectors):
    return normalize(vectors, axis=1)

def chunk_text(text, chunk_size=512):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def cache_embeddings(docs, embedder, cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    else:
        embeddings = embedder.embed_documents(docs)
        embeddings = normalize_vectors(np.array(embeddings))
        with open(cache_path, "wb") as f:
            pickle.dump(embeddings, f)
        return embeddings

def save_faiss_index(index, path):
    index.save_local(path)

def load_faiss_index(path, embedder):
    return FAISS.load_local(path, embedder)

def rerank_docs(query, docs, model_name, top_k=5):
    cross_encoder = CrossEncoder(model_name)
    pairs = [[query, doc] for doc in docs]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, docs), reverse=True)
    return [doc for _, doc in ranked[:top_k]]

def list_faiss_indices():
    return [f for f in os.listdir(INDEX_DIR) if os.path.isdir(os.path.join(INDEX_DIR, f))]

# --- Interface ---
with st.sidebar:
    st.header("Configurações do Modelo")
    llm_choice = st.selectbox("LLM", ["Ollama", "GPT-4", "Copilot"])
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.7)
    reranker_model = st.selectbox("Reranker", ["BAAI/bge-reranker-large", "cross-encoder/ms-marco-MiniLM-L-6-v2"])
    embedding_model = st.selectbox("Modelo de Embedding", ["multilingual_e5_large", "bge-base-pt"])
    chunk_size = st.slider("Tamanho do chunk (palavras)", 128, 1024, 512, step=64)
    top_k_retrieval = st.slider("Top-K recuperação", 3, 20, 10)
    top_k_rerank = st.slider("Top-K reranking", 1, 10, 5)
    index_version = st.text_input("Versão do índice (opcional)", value=datetime.now().strftime("%Y%m%d_%H%M%S"))
    available_indices = list_faiss_indices()
    selected_indices = st.multiselect("Índices FAISS disponíveis", available_indices)

uploaded_files = st.file_uploader("Carregar CSVs", type="csv", accept_multiple_files=True)

# --- Processamento e Indexação ---
def process_and_index(files, embedding_model, chunk_size, index_version):
    all_chunks = []
    for file in files:
        df = pd.read_csv(file)
        for _, row in df.iterrows():
            # Exemplo: ajuste para os nomes das colunas do seu CSV
            texto = f"Programa {row['CODIGO']}: {row['OBJETIVO']}\nMetas: {row['METAS']}"
            chunks = chunk_text(texto, chunk_size)
            if embedding_model == "multilingual_e5_large":
                chunks = [format_doc_e5(chunk) for chunk in chunks]
            all_chunks.extend(chunks)
    # Embeddings e cache
    cache_path = os.path.join(EMBEDDING_CACHE_DIR, f"{embedding_model}_{index_version}.pkl")
    if embedding_model == "multilingual_e5_large":
        embedder = OllamaEmbeddings(model="multilingual_e5_large")
    else:
        from langchain_huggingface import HuggingFaceEmbeddings

        embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-base-pt")
    embeddings = cache_embeddings(all_chunks, embedder, cache_path)
    # Criação do índice FAISS
    index = FAISS.from_embeddings(all_chunks, embeddings, embedder)
    index_path = os.path.join(INDEX_DIR, f"{embedding_model}_{index_version}")
    save_faiss_index(index, index_path)
    return index_path

if st.button("Indexar Dados"):
    if uploaded_files:
        with st.spinner("Processando e indexando..."):
            index_path = process_and_index(uploaded_files, embedding_model, chunk_size, index_version)
        st.success(f"Indexação concluída! Índice salvo em: {index_path}")

# --- Chat e Recuperação ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Faça sua pergunta sobre os programas:"):
    retrieved_chunks = []
    # Recupera de todos os índices selecionados
    for idx in selected_indices:
        idx_path = os.path.join(INDEX_DIR, idx)
        if embedding_model == "multilingual_e5_large":
            embedder = OllamaEmbeddings(model="multilingual_e5_large")
            query = format_query_e5(prompt)
        else:
            from langchain_huggingface import HuggingFaceEmbeddings

            embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-base-pt")
            query = prompt
        index = load_faiss_index(idx_path, embedder)
        docs = index.similarity_search(query, k=top_k_retrieval)
        retrieved_chunks.extend([d.page_content for d in docs])
    # Reranking
    reranked = rerank_docs(prompt, retrieved_chunks, reranker_model, top_k=top_k_rerank)
    context = "\n\n".join(reranked)
    prompt_template = f"""
    Contexto:
    {context}

    Pergunta: {prompt}
    Resposta:
    """
    # Chamada ao LLM (exemplo com Ollama)
    if llm_choice == "Ollama":
        import ollama
        response = ollama.generate(
            model="llama3.2",
            prompt=prompt_template,
            temperature=temperature
        )["response"]
    elif llm_choice == "GPT-4":
        import openai
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "Você é um assistente especialista em programas plurianuais."},
                      {"role": "user", "content": prompt_template}],
            temperature=temperature
        )["choices"][0]["message"]["content"]
    else:
        response = "Integração com Copilot não implementada neste exemplo."
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})

# --- Logging básico ---
if st.checkbox("Mostrar histórico de logs"):
    st.write(st.session_state.messages)
