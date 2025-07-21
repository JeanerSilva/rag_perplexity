import streamlit as st
from config.settings import INDEX_DIR, DATA_FOLDER
from utils.index_utils import save_index_bundle
from utils.embedding_utils import get_embedder
from components.sidebar import show_sidebar
from components.chat_handler import handle_chat
from langchain_core.documents import Document
from utils.process_pdf import process_pdf_to_jsonl
from utils.file_utils import read_jsonl_files


# Configuração da página
st.set_page_config(page_title="Pergunte ao PPA", layout="wide")

# UI lateral
options = show_sidebar()

uploaded_files = st.file_uploader("Carregar CSVs", type="csv", accept_multiple_files=True) if options['data_source'] == "Upload manual" else None

# Indexação
uploaded_pdf = st.file_uploader("Carregar PDF do programa", type="pdf")


# UI para upload
uploaded_pdf = st.file_uploader("Carregar PDF do programa", type="pdf", key="pdf_upload_main") if options["data_source"] == "Upload manual" else None

if st.button("Iniciar Indexação"):
    try:
        with st.spinner("Processando e indexando..."):

            # 📁 Escolha do caminho do PDF
            if options["data_source"] == "Pasta padrão":
                pdf_path = "fonte_de_dados/Espelho_SIOP_1752869188991.pdf"
                st.write(f"📁 Usando PDF local salvo: `{pdf_path}`")
            elif uploaded_pdf is not None:
                pdf_path = f"/tmp/{uploaded_pdf.name}"
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_pdf.read())
                st.write(f"📤 PDF recebido por upload: `{pdf_path}`")
            else:
                st.warning("⚠️ Você deve carregar um PDF.")
                st.stop()

            # 📄 Gera JSONL com chunks do PDF
            st.write("📦 Extraindo informações e gerando chunks...")
            jsonl_path, total_chunks = process_pdf_to_jsonl(
                pdf_path,
                "fonte_de_dados/dados_abertos/programas_chunked_tokenlimit_300.jsonl",
                token_limit=options["chunk_size"]
            )

            st.write(f"✅ JSONL gerado: `{jsonl_path}` com `{total_chunks}` chunks")

            # 📚 Carregar os dados do JSONL
            #raw_texts = read_jsonl_files([jsonl_path], options["chunk_size"], options["embedding_model"])
            raw_texts = read_jsonl_files(["fonte_de_dados/dados_abertos/programas_chunked_tokenlimit_300_jupiter.jsonl"], options["chunk_size"], options["embedding_model"])

            # 🔍 Log de debug dos primeiros chunks
            st.write("🔍 Visualização dos primeiros chunks:")
            for i, item in enumerate(raw_texts[:5]):
                st.markdown(f"**Chunk {i+1}**")
                st.markdown(f"`campos_presentes:` `{item['metadata'].get('campos_presentes')}`")
                st.code(item["text"][:500] + ("..." if len(item["text"]) > 500 else ""))

            documents = [
                Document(page_content=doc["text"], metadata=doc["metadata"])
                for doc in raw_texts
            ]

            st.write(f"🧠 Total de documentos convertidos para indexação: {len(documents)}")

            # 💾 Embedding e indexação
            embedder = get_embedder(options["embedding_model"])
            index_name = f"{options['embedding_model']}_{options['index_version']}"
            save_index_bundle(documents, embedder, index_name)

            st.success(f"🎯 Indexação concluída com sucesso: `{index_name}`")
            st.write(f"📊 Total de chunks indexados: {len(documents)}")
            st.write(f"🧷 Campos detectados nos primeiros metadados: {[doc.metadata.get('campos_presentes') for doc in documents[:5]]}")

    except Exception as e:
        st.error(f"❌ Erro durante a indexação: {e}")
        import traceback
        st.code(traceback.format_exc())



# Chat
handle_chat(options)
