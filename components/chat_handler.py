import streamlit as st
from utils.embedding_utils import get_embedder, format_query_e5
from utils.index_utils import load_index_bundle
from sentence_transformers import CrossEncoder
import ollama
from openai import OpenAI


def inferir_tipos_relevantes(pergunta):
    pergunta = pergunta.lower()
    tipos = []

    if "objetivo específico" in pergunta or "objetivos específicos" in pergunta:
        tipos.append("objetivo_especifico")
    if "objetivo geral" in pergunta:
        tipos.append("objetivo_geral")
    if "objetivo estratégico" in pergunta or "objetivos estratégicos" in pergunta:
        tipos.append("objetivo_estrategico")
    if "problema" in pergunta:
        tipos.append("problema")
    if "causa" in pergunta:
        tipos.append("causa")
    if "justificativa" in pergunta:
        tipos.append("justificativa")
    if "evidência" in pergunta or "evidencias" in pergunta:
        tipos.append("evidencia")
    if "entrega" in pergunta or "entregas" in pergunta:
        tipos.append("entrega")
    if "público-alvo" in pergunta or "público alvo" in pergunta:
        tipos.append("publico_alvo")

    return tipos or None


def handle_chat(options):
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Renderiza mensagens anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrada do usuário
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
            retrieved_chunks.extend(results)  # Mantém como `Document` para preservar metadados

        # Filtro por tipo, baseado na pergunta
        tipos_desejados = inferir_tipos_relevantes(prompt)
        if tipos_desejados:
            retrieved_chunks = [
                doc for doc in retrieved_chunks
                if doc.metadata.get("tipo") in tipos_desejados
            ]

        if not retrieved_chunks:
            st.warning("Nenhum chunk relevante encontrado com base nos filtros aplicados.")
            return

        # Apresenta os chunks recuperados inicialmente
        with st.expander("🔎 Chunks Recuperados (antes do reranking)", expanded=False):
            for i, doc in enumerate(retrieved_chunks, 1):
                st.markdown(f"**Chunk {i}:**")
                st.markdown(f"`Metadados:` `{doc.metadata}`")
                st.markdown(f"```\n{doc.page_content}\n```")

        # Rerank
        cross_encoder = CrossEncoder(options['reranker_model'])
        pairs = [[prompt, doc.page_content] for doc in retrieved_chunks]
        scores = cross_encoder.predict(pairs)
        ranked = sorted(zip(scores, retrieved_chunks), reverse=True)
        reranked_chunks = [doc.page_content for score, doc in ranked[:options['top_k_rerank']]]

        # Mostrar após rerank
        with st.expander("🏆 Chunks Selecionados após Reranking", expanded=True):
            for i, (score, doc) in enumerate(ranked[:options['top_k_rerank']], 1):
                st.markdown(f"**#{i} — Score: {score:.4f}**")
                st.markdown(f"`Metadados:` `{doc.metadata}`")
                st.markdown(f"```\n{doc.page_content}\n```")

        context = "\n\n".join(reranked_chunks)

        # Prompt do sistema
        system_prompt = (
            "Você é um assistente especializado na análise de documentos de planejamento público, com acesso a trechos "
            "documentos relativos ao Plano Plurianual (PPA).\n"
            "Seu trabalho é responder com base **exclusivamente no conteúdo abaixo**, sem usar conhecimento externo ou fazer suposições.\n\n"
            f"📄 **Trechos do documento (contexto):**\n{context}\n\n"
            f"❓ **Pergunta:**\n{prompt}\n\n"
            "📌 **Instruções de resposta**:\n"
            "- Utilize os metadados dos chunks para identificar claramente o tipo de informação (ex: objetivo específico, problema, justificativa...)\n"
            "- Liste sempre todos os que se referem ao mesmo programa, sem exceção.\n"
            "- Se a resposta estiver nos trechos, **repita exatamente a mesma redação**.\n"
            "- Nunca misture conceitos como objetivo geral, estratégico e específico.\n"
            "- Ignore qualquer informação fora dos trechos. Responda com base **exclusiva** no que foi extraído.\n"
            "🔁 Agora responda:"
        )

        # Escolha do modelo
        llm = options["llm_choice"]

        if llm == "GPT-4":
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context + "\n\n" + prompt}
                ],
                temperature=options["temperature"]
            )
            response_text = response.choices[0].message.content

        elif llm == "Ollama":
            response = ollama.chat(
                model="llama3",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context + "\n\n" + prompt}
                ]
            )
            response_text = response["message"]["content"]

        elif llm == "Copilot":
            response_text = "⚠️ O modelo Copilot ainda não foi implementado."

        # Mostra resposta
        with st.chat_message("assistant"):
            st.markdown(response_text)

        # Histórico
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response_text})
