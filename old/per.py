import streamlit as st
import pandas as pd
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

# Configuração inicial
st.set_page_config(page_title="Analisador PLURIANUAL", layout="wide")

# Componentes da UI
with st.sidebar:
    st.header("Configurações do Modelo")
    llm_choice = st.selectbox("LLM", ["Ollama", "GPT-4", "Copilot"])
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.7)
    reranker_model = st.selectbox("Reranker", ["bge-reranker-large", "CohereRerank"])
    faiss_models = st.multiselect("Modelos FAISS", ["multilingual_e5_large", "bge-base-pt"])

# Carregamento de dados
uploaded_files = st.file_uploader("Carregar CSVs", type="csv", accept_multiple_files=True)

# Processamento dos dados
@st.cache_resource
def process_data(files):
    all_docs = []
    for file in files:
        df = pd.read_csv(file)
        for _, row in df.iterrows():
            doc = f"Programa {row['CODIGO']}: {row['OBJETIVO']}\nMetas: {row['METAS']}"
            all_docs.append(doc)
    return all_docs

# Indexação FAISS
if st.button("Indexar Dados"):
    with st.spinner("Processando..."):
        documents = process_data(uploaded_files)
        
        # Cria múltiplos índices
        vector_stores = {}
        if "multilingual_e5_large" in faiss_models:
            embeddings = OllamaEmbeddings(model="multilingual_e5_large")
            vector_stores["e5"] = FAISS.from_texts(documents, embeddings)
        
        if "bge-base-pt" in faiss_models:
            embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-pt")
            vector_stores["bge"] = FAISS.from_texts(documents, embeddings)
        
        st.session_state.vector_stores = vector_stores
    st.success("Indexação concluída!")

# Sistema de Reranking
def rerank_docs(query, docs, model_name):
    cross_encoder = CrossEncoder(model_name)
    pairs = [[query, doc] for doc in docs]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, docs), reverse=True)
    return [doc for _, doc in ranked[:5]]

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Faça sua pergunta sobre os programas:"):
    # Recuperação Multi-Índice
    all_docs = []
    for vs in st.session_state.vector_stores.values():
        docs = vs.similarity_search(prompt, k=3)
        all_docs.extend([d.page_content for d in docs])
    
    # Reranking
    reranked = rerank_docs(prompt, all_docs, reranker_model)
    
    # Geração da Resposta
    context = "\n\n".join(reranked)
    prompt_template = f"""
    Contexto:
    {context}

    Pergunta: {prompt}
    Resposta:
    """
    
    # Seleção do LLM
    if llm_choice == "Ollama":
        response = ollama.generate(
            model="llama3.2",
            prompt=prompt_template,
            temperature=temperature
        )
    elif llm_choice == "GPT-4":
        response = openai.ChatCompletion.create(...)
    
    # Exibição
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
