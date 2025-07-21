import streamlit as st
from utils.embedding_utils import get_embedder, format_query_e5
from utils.index_utils import load_index_bundle
from sentence_transformers import CrossEncoder
import ollama
from openai import OpenAI
import re
import json

def extrair_programa_da_pergunta(pergunta):
    """
    Extrai o código e o nome do programa a partir da pergunta do usuário.
    Retorna: (programa_codigo, programa_nome)
    """
    codigo = None
    nome = None

    match_codigo = re.search(r'programa\s+(\d+)', pergunta.lower())
    if match_codigo:
        codigo = match_codigo.group(1)

    match_nome = re.search(r'programa\s+(?:\d+\s*[-–]\s*)?(.+?)(?:[?.!]|$)', pergunta, re.IGNORECASE)
    if match_nome:
        nome = match_nome.group(1).strip().lower()

    return codigo, nome

def inferir_tipos_relevantes_regex(pergunta):
    pergunta = pergunta.lower()
    tipos = []

    if re.search(r"programa", pergunta):
        tipos.append("programa_codigo") 
    if re.search(r"\bevolu[çc][ãa]o hist[óo]rica\b", pergunta):
        tipos.append("evolucao_historica")   
    if re.search(r"programa", pergunta):
        tipos.append("programa_nome")
    if re.search(r"\bobjetiv[oa]s?\s+específic[oa]s?\b", pergunta):
        tipos.append("objetivo_especifico")
    if re.search(r"\bobjetiv[oa]s?\s+gerais?\b", pergunta):
        tipos.append("objetivo_geral")
    if re.search(r"\bobjetiv[oa]s?\s+estratégic[oa]s?\b", pergunta):
        tipos.append("objetivo_estrategico")
    if re.search(r"\bproblema\b|\bproblemas?\b", pergunta):
        tipos.append("evidências_do_problema")
    if re.search(r"\bcausa\b|\bcausas\b", pergunta):
        tipos.append("causa")
    if re.search(r"\bjustificativa\b|\bpor que\b|\bporque\b|\binterven[çc][ãa]o\b", pergunta):
        tipos.append("justificativa_para_a_intervenção")
    if re.search(r"\bevid[êe]ncia\b|\bevid[êe]ncias\b", pergunta):
        tipos.append("evidências_do_problema")
    if re.search(r"\bentregas?\b|\bresultados?\b", pergunta):
        tipos.append("entrega")
    if re.search(r"\bp[úu]blico[- ]?alvo\b", pergunta):
        tipos.append("publico_alvo")
    if re.search(r"\bmarco legal\b", pergunta):
        tipos.append("marco_legal")
    if re.search(r"\barticula[çc][ãa]o federativa\b", pergunta):
        tipos.append("articulacao_federativa")
    if re.search(r"\bcompara[çc][ãa]o internacional\b", pergunta):
        tipos.append("comparações_internacionais")
    if re.search(r"\brela[çc][ãa]o com ODS\b", pergunta):
        tipos.append("relação_com_ODS")
    if re.search(r"\benfoque transvers[alis]\b", pergunta):
        tipos.append("enfoque_transversal")
    if re.search(r"\bmarco[s] lega[alis]\b", pergunta):
        tipos.append("marco_legal")

        

    return tipos or None

def inferir_tipos_com_llm(pergunta):
    try:
        client = OpenAI()
        system_prompt = (
            "Você é um classificador. Dada uma pergunta sobre planejamento público, identifique qual tipo de conteúdo ela solicita "
            "dentre as seguintes categorias:\n\n"
            "- objetivo_especifico\n"
            "- objetivo_geral\n"
            "- objetivo_estrategico\n"
            "- problema\n"
            "- causa\n"
            "- justificativa\n"
            "- evidencia\n"
            "- entrega\n"
            "- publico_alvo\n"
            "- marco_legal\n"
            "- articulacao_federativa\n\n"
            "Retorne apenas uma lista JSON com as chaves exatas, por exemplo: [\"problema\", \"justificativa\"]"
        )
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": pergunta}
            ],
            temperature=0
        )
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            st.warning("⚠️ A resposta da LLM não pôde ser convertida em JSON. Nenhum tipo filtrado será aplicado.")
            return None

    except Exception as e:
        print(f"⚠️ Erro ao inferir tipo com LLM: {e}")
        return None

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
            print(f"🔎 [DEBUG] {idx} — {len(results)} chunks recuperados.")
            for i, r in enumerate(results):
                print(f"  → Chunk {i+1}: campos_presentes={r.metadata.get('campos_presentes')}, metadados={r.metadata}")
            retrieved_chunks.extend(results)

        with st.expander("🧠 Tipos inferidos com base na pergunta", expanded=False):
            tipos_regex = inferir_tipos_relevantes_regex(prompt)
            st.write("🔍 Detecção via regex:", tipos_regex)

            tipos_desejados = tipos_regex

            if not tipos_desejados and options["llm_choice"] == "GPT-4":
                st.write("⚠️ Nenhum tipo identificado por regex. Chamando LLM...")
                tipos_desejados = inferir_tipos_com_llm(prompt)
                st.write("🤖 Tipos sugeridos pela LLM:", tipos_desejados)

        st.write("🔍 Tipos usados para filtragem:", tipos_desejados)
        print(f"🔍 [DEBUG] Tipos desejados: {tipos_desejados}")
        print(f"📄 [DEBUG] Total de chunks antes do filtro: {len(retrieved_chunks)}")

        codigo_desejado, nome_desejado = extrair_programa_da_pergunta(prompt)

        for i, doc in enumerate(retrieved_chunks, 1):
            print(f"  → Chunk {i}: campos_presentes={doc.metadata.get('campos_presentes')}")

        if tipos_desejados:
          retrieved_chunks = [
                doc for doc in retrieved_chunks
                if any(t in doc.metadata.get("campos_presentes", []) for t in tipos_desejados)
                and (
                    (not codigo_desejado or doc.metadata.get("programa_codigo") == codigo_desejado)
                    or (nome_desejado and nome_desejado in doc.metadata.get("programa_nome", "").lower())
                )
          ]

        print(f"📄 [DEBUG] Total de chunks após filtro: {len(retrieved_chunks)}")
        if retrieved_chunks:
            for i, doc in enumerate(retrieved_chunks, 1):
                print(f"  ✅ Chunk {i} passou no filtro — campos_presentes: {doc.metadata.get('campos_presentes')}")
        else:
            print("⚠️ [DEBUG] Nenhum chunk passou no filtro. Verifique tipos ou metadados.")

        if not retrieved_chunks:
            st.warning("Nenhum chunk relevante encontrado com base nos filtros aplicados.")
            return

        with st.expander("🔎 Chunks Recuperados (antes do reranking)", expanded=False):
            for i, doc in enumerate(retrieved_chunks, 1):
                st.markdown(f"**Chunk {i}:**")
                st.markdown(f"`Metadados:` `{doc.metadata}`")
                st.markdown(f"```\n{doc.page_content}\n```")

        cross_encoder = CrossEncoder(options['reranker_model'])
        pairs = [[prompt, doc.page_content] for doc in retrieved_chunks]
        scores = cross_encoder.predict(pairs)
        ranked = sorted(zip(scores, retrieved_chunks), key=lambda x: x[0], reverse=True)
        reranked_chunks = [doc.page_content for score, doc in ranked[:options['top_k_rerank']]]

        with st.expander("🏆 Chunks Selecionados após Reranking", expanded=True):
            for i, (score, doc) in enumerate(ranked[:options['top_k_rerank']], 1):
                st.markdown(f"**#{i} — Score: {score:.4f}**")
                st.markdown(f"`Metadados:` `{doc.metadata}`")
                st.markdown(f"```\n{doc.page_content}\n```")

        context = "\n\n".join(reranked_chunks)

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

        with st.chat_message("assistant"):
            st.markdown(response_text)

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response_text})

        with st.expander("🧾 Resumo do processamento", expanded=False):
            st.markdown(f"- **Tipos usados no filtro:** `{tipos_desejados or 'Nenhum (todos considerados)'}`")
            st.markdown(f"- **Chunks reranqueados:** {len(reranked_chunks)}")
