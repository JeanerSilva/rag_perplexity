import streamlit as st
from utils.embedding_utils import get_embedder, format_query_e5
from utils.index_utils import load_index_bundle
from utils.rerank_utils import rerank_docs

def handle_chat(options):
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Faça sua pergunta sobre os programas:"):
        retrieved_chunks = []

        if not options["selected_indices"]:
            st.warning("Selecione pelo menos um índice.")
            return

        embedder = get_embedder(options['embedding_model'])
        query = format_query_e5(prompt) if options['embedding_model'] == "multilingual_e5_large" else prompt

        for idx in options['selected_indices']:
            index = load_index_bundle(idx, embedder)
            results = index.similarity_search(query, k=options["top_k_retrieval"])
            retrieved_chunks.extend([doc.page_content for doc in results])

        reranked = rerank_docs(prompt, retrieved_chunks, options['reranker_model'], top_k=options['top_k_rerank'])
        context = "\n\n".join(reranked)

        # Exemplo com Ollama — adapte conforme necessário
        import ollama
        response = ollama.chat(
            model="llama3",
            messages=[
                {"role": "system", "content": "Você é um assistente especialista em programas plurianuais."},
                {"role": "user", "content": context + "\n\n" + prompt}
            ]
        )
        response_text = response["message"]["content"]

        with st.chat_message("assistant"):
            st.markdown(response_text)

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response_text})
