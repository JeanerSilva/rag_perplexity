import streamlit as st
import pandas as pd
import os
import pickle
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from sklearn.preprocessing import normalize
import numpy as np
import faiss
from datetime import datetime
import glob
import csv

from langchain_core.documents import Document

def to_langchain_documents(chunks):
    return [Document(page_content=chunk) for chunk in chunks]

# --- Configurações Globais ---
st.set_page_config(page_title="Pergunte ao PPA", layout="wide")
DATA_FOLDER = "fonte_de_dados/dados_abertos"
INDEX_DIR = "faiss_indices"
EMBEDDING_CACHE_DIR = "embedding_cache"
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)

# --- Funções Auxiliares ---
def format_query_e5(text): return "query: " + text
def format_doc_e5(text): return "passage: " + text

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

def cache_embeddings(docs, embedder):
    if len(docs) == 0:
        raise ValueError("Lista de documentos está vazia.")
    embeddings = embedder.embed_documents(docs)
    if embeddings is None or len(embeddings) == 0:
        raise ValueError("Nenhum embedding foi retornado.")
    embeddings = np.array(embeddings)
    embeddings = normalize_vectors(embeddings)
    return embeddings

def create_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index

# --- Funções Atualizadas de Salvamento/Carregamento ---
def save_index_bundle(documents, embedder, index_name):
    index_dir = os.path.join(INDEX_DIR, index_name)
    os.makedirs(index_dir, exist_ok=True)

    # Cria o índice com documentos e embedder (FAISS calcula embeddings internamente)
    vectorstore = FAISS.from_documents(documents, embedder)

    # Salva no diretório
    vectorstore.save_local(index_dir)

def load_index_bundle(index_name, embedder):
    index_dir = os.path.join(INDEX_DIR, index_name)
    try:
        vectorstore = FAISS.load_local(index_dir, embedder, allow_dangerous_deserialization=True)
        return vectorstore
    except Exception as e:
        st.error(f"Erro ao carregar índice: {e}")
        return None

def rerank_docs(query, docs, model_name, top_k=5):
    cross_encoder = CrossEncoder(model_name)
    pairs = [[query, doc] for doc in docs]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, docs), reverse=True)
    return [doc for _, doc in ranked[:top_k]]

def list_index_names():
    return [name for name in os.listdir(INDEX_DIR) if os.path.isdir(os.path.join(INDEX_DIR, name))]

# --- Interface do Usuário ---
with st.sidebar:
    st.header("Configurações do Modelo")
    data_source = st.radio("Fonte de dados", ["Pasta padrão", "Upload manual"])
    llm_choice = st.selectbox("LLM", ["Ollama", "GPT-4", "Copilot"])
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.7)
    reranker_model = st.selectbox("Reranker", ["BAAI/bge-reranker-large", "cross-encoder/ms-marco-MiniLM-L-6-v2"])
    embedding_model = st.selectbox("Modelo de Embedding", ["multilingual_e5_large", "bge-base-pt"])
    chunk_size = st.slider("Tamanho do chunk (palavras)", 128, 1024, 512, step=64)
    top_k_retrieval = st.slider("Top-K recuperação", 3, 20, 10)
    top_k_rerank = st.slider("Top-K reranking", 1, 10, 5)
    index_version = st.text_input("Versão do índice", value=datetime.now().strftime("%Y%m%d_%H%M%S"))
    selected_indices = st.multiselect("Índices disponíveis", list_index_names())

uploaded_files = st.file_uploader("Carregar CSVs", type="csv", accept_multiple_files=True) if data_source == "Upload manual" else None

# --- Processamento de Dados ---
def process_documents(source, files=None):
    if source == "Pasta padrão":
        documents = read_csvs_from_folder(DATA_FOLDER)
    else:
        documents = []
        for file in files:
            df = pd.read_csv(file)
            for _, row in df.iterrows():
                texto = f"Programa {row['CODIGO']}: {row['OBJETIVO']}\nMetas: {row['METAS']}"
                chunks = chunk_text(texto, chunk_size)
                if embedding_model == "multilingual_e5_large":
                    chunks = [format_doc_e5(chunk) for chunk in chunks]
                documents.extend(chunks)
    return documents

# --- Indexação ---
if st.button("Iniciar Indexação"):
    try:
        with st.spinner("Processando e indexando..."):
            raw_texts = process_documents(data_source, uploaded_files)
            if not raw_texts:
                raise ValueError("Nenhum documento encontrado!")

            # Converte para Document
            documents = to_langchain_documents(raw_texts)

            # Cria embedder correto
            if embedding_model == "multilingual_e5_large":
                embedder = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
            else:
                embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-base-pt")

            # Cria e salva índice FAISS
            index_name = f"{embedding_model}_{index_version}"
            save_index_bundle(documents, embedder, index_name)

            st.success(f"Indexação concluída: {index_name}")
            st.write(f"Total de chunks indexados: {len(documents)}")

    except Exception as e:
        st.error(f"Erro durante a indexação: {e}")


# --- Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Faça sua pergunta sobre os programas:"):
    retrieved_chunks = []

    if not selected_indices:
        st.warning("Selecione pelo menos um índice.")
    else:
        for idx_name in selected_indices:
            embedder = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large") if embedding_model == "multilingual_e5_large" else HuggingFaceEmbeddings(model_name="BAAI/bge-base-pt")
            query = format_query_e5(prompt) if embedding_model == "multilingual_e5_large" else prompt

            index = load_index_bundle(idx_name, embedder)
            if index is None:
                continue

            search_results = index.similarity_search(query, k=top_k_retrieval)
            retrieved_chunks.extend([doc.page_content for doc in search_results])

        reranked = rerank_docs(prompt, retrieved_chunks, reranker_model, top_k=top_k_rerank)
        context = "\n\n".join(reranked)

        prompt_template = f"""
        Contexto:
        {context}

        Pergunta: {prompt}
        Resposta:
        """

        if llm_choice == "Ollama":
            import ollama

            response = ollama.chat(
                model="llama3",
                messages=[
                    {"role": "system", "content": "Você é um assistente especializado na análise de documentos de planejamento público, com acesso a trechos documentos relativos ao Plano Plurianual (PPA).\nSeu trabalho é responder com base **exclusivamente no conteúdo abaixo**, sem usar conhecimento externo ou fazer suposições.\n\n📄 **Trechos do documento (contexto):**\n{context}\n\n❓ **Pergunta:**\n{question}\n \n📌 **Instruções de resposta**:\n\n- O conteúdo está no formato de jsonl. E cada linha contem metadados que indicam categorias como \"objetivo_geral\", \"objetivos_especificos\", \"objetivos_estrategicos\", \"publico_alvo\" e \"orgao_responsavel\" cada um deles se referindo a um \"programa_id\".\n\nSe houver pergunta que te leve a fazer uma lista, liste sempre todos os que se referem ao mesmo programa, sem exceção.\nPor exemplo, se for perguntado quais os objetivos estratégicos de um programa, como por exemplo 1144, todas linhas com a categoria \"objetivos_estrategicos\" com o programa \"programa_id\" 1144 devem ser listadas. Nenhuma pode não ser citada.\n- Se a resposta for objetiva e identificável nos trechos, **repita exatamente a mesma redação todas as vezes**. Não interrompa uma lista. Vá até o final.\n- Se algo não for pedido, não cite.\n- Se não forem pedidos objetivos específicos não cite. \n- Se a lista for longa, continue até o final, sem interromper ou resumir.\n- Considere o valor dos metadados de categoria para responder.\n- Se houver vários itens, liste todos os itens com uma lista numerada.\"\n- Não confunda os conceitos de \"gerais\", \"estratégicos\" e \"específicos\". Cada um deles deve ser tratado especificamente conforme o contexto defina.\n🔁 Agora responda:"},
                    {"role": "user", "content": prompt_template}
                ]
            )
            response_text = response['message']['content']
        elif llm_choice == "GPT-4":
            import openai
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Você é um assistente especialista em programas plurianuais."},
                    {"role": "user", "content": prompt_template}
                ],
                temperature=temperature
            )["choices"][0]["message"]["content"]
        else:
            response = "Integração com Copilot não implementada."

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- Logs ---
if st.checkbox("Mostrar histórico de logs"):
    st.write(st.session_state.messages)
