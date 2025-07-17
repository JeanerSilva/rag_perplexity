import streamlit as st
from config.settings import DEFAULT_INDEX_VERSION
from utils.index_utils import list_index_names

def show_sidebar():
    with st.sidebar:
        st.header("Configurações do Modelo")
        return {
            "data_source": st.radio("Fonte de dados", ["Pasta padrão", "Upload manual"]),
            "llm_choice": st.selectbox("LLM", ["GPT-4", "Ollama", "Copilot"]),
            "temperature": st.slider("Temperatura", 0.0, 1.0, 0.7),
            "reranker_model": st.selectbox("Reranker", ["BAAI/bge-reranker-large", "cross-encoder/ms-marco-MiniLM-L-6-v2"]),
            "embedding_model": st.selectbox("Modelo de Embedding", ["multilingual_e5_large", "bge-base-pt"]),
            "chunk_size": st.slider("Tamanho do chunk (palavras)", 128, 1024, 256, step=64),
            "top_k_retrieval": st.slider("Top-K recuperação", 3, 60, 20),
            "top_k_rerank": st.slider("Top-K reranking", 1, 20, 10),
            "index_version": st.text_input("Versão do índice", value=DEFAULT_INDEX_VERSION),
            "selected_indices": st.multiselect("Índices disponíveis", list_index_names())
        }
