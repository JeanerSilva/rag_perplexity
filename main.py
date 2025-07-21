import streamlit as st
from config.settings import INDEX_DIR, DATA_FOLDER
from utils.index_utils import save_index_bundle
from utils.embedding_utils import get_embedder
from components.sidebar import show_sidebar
from components.chat_handler import handle_chat
from langchain_core.documents import Document
from utils.process_pdf import process_pdf_to_jsonl
from utils.file_utils import read_jsonl_files


uploaded_pdf = st.file_uploader("Carregar PDF do programa", type="pdf")

# Configuração da página
st.set_page_config(page_title="Pergunte ao PPA", layout="wide")

# UI lateral
options = show_sidebar()

uploaded_files = st.file_uploader("Carregar CSVs", type="csv", accept_multiple_files=True) if options['data_source'] == "Upload manual" else None

# Indexação
uploaded_pdf = st.file_uploader("Carregar PDF do programa", type="pdf")

from process_pdf import process_pdf_to_jsonl
from utils.file_utils import read_jsonl_files
from langchain_core.documents import Document

# UI
uploaded_pdf = st.file_uploader("Carregar PDF do programa", type="pdf") if options["data_source"] == "Upload manual" else None

if st.button("Iniciar Indexação"):
    try:
        with st.spinner("Processando e indexando..."):

            # 🔁 Caminho para o PDF dependendo da fonte
            if options["data_source"] == "Pasta padrão":
                pdf_path = "fonte_de_dados/Espelho_SIOP_1752869188991.pdf"  # ou outro nome fixo que você usa
            elif uploaded_pdf is not None:
                pdf_path = f"/tmp/{uploaded_pdf.name}"
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_pdf.read())
            else:
                st.warning("⚠️ Você deve carregar um PDF.")
                st.stop()

            # 📄 Gera JSONL com os chunks do PDF
            jsonl_path, total_chunks = process_pdf_to_jsonl(
                pdf_path,
                "programas_chunked_tokenlimit_300.jsonl",
                token_limit=options["chunk_size"]
            )

            # 📚 Lê chunks e transforma em Document
            raw_texts = read_jsonl_files([jsonl_path], options["chunk_size"], options["embedding_model"])
            documents = [
                Document(page_content=doc["text"], metadata=doc["metadata"])
                for doc in raw_texts
            ]

            # 💾 Embedding + Indexação
            embedder = get_embedder(options["embedding_model"])
            index_name = f"{options['embedding_model']}_{options['index_version']}"
            save_index_bundle(documents, embedder, index_name)

            # ✅ Feedback
            st.success(f"Indexação concluída: {index_name}")
            st.write(f"📄 Total de chunks: {len(documents)}")
            st.write(f"🔎 Fonte de dados: `{options['data_source']}`")
            st.write(f"📁 PDF processado: `{pdf_path}`")

    except Exception as e:
        st.error(f"Erro durante a indexação: {e}")



# Chat
handle_chat(options)
