import streamlit as st
from config.settings import INDEX_DIR, DATA_FOLDER
from utils.file_utils import process_documents
from utils.index_utils import save_index_bundle, list_index_names
from utils.embedding_utils import get_embedder, chunk_text
from utils.rerank_utils import rerank_docs
from components.sidebar import show_sidebar
from components.chat_handler import handle_chat
from langchain_core.documents import Document

# Configuração da página
st.set_page_config(page_title="Pergunte ao PPA", layout="wide")

# UI lateral
options = show_sidebar()

uploaded_files = st.file_uploader("Carregar CSVs", type="csv", accept_multiple_files=True) if options['data_source'] == "Upload manual" else None

# Indexação
if st.button("Iniciar Indexação"):
    try:
        with st.spinner("Processando e indexando..."):
            raw_texts = process_documents(options['data_source'], uploaded_files, options)
            documents = []

            for item in raw_texts:
                texto_original = item["text"]
                metadados = item.get("metadata", {})
                chunks = chunk_text(texto_original, options["chunk_size"])
                
                for i, chunk in enumerate(chunks):
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={**metadados, "chunk_id": i + 1}
                        )
                    )

            embedder = get_embedder(options['embedding_model'])
            index_name = f"{options['embedding_model']}_{options['index_version']}"
            save_index_bundle(documents, embedder, index_name)
            st.success(f"Indexação concluída: {index_name}")
            st.write(f"Total de chunks indexados: {len(documents)}")
    except Exception as e:
        st.error(f"Erro durante a indexação: {e}")

# Chat
handle_chat(options)
