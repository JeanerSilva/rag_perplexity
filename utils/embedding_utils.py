from langchain_huggingface import HuggingFaceEmbeddings

def get_embedder(model_name):
    if model_name == "multilingual_e5_large":
        return HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    return HuggingFaceEmbeddings(model_name="BAAI/bge-base-pt")


def format_doc_e5(text):
    return "passage: " + text


def format_query_e5(text):
    return "query: " + text


def chunk_text(text, chunk_size):
    """
    Divide o texto em chunks lógicos baseados no delimitador `;`,
    sem ultrapassar o número máximo de palavras definido em `chunk_size`.

    - Mantém unidades semânticas agrupadas (ex: uma causa ou entrega).
    - Retorna uma lista de strings, cada uma com até `chunk_size` palavras.
    """

    # Limpeza inicial
    unidades = [u.strip() for u in text.replace("\n", " ").split(";") if u.strip()]
    
    chunks = []
    chunk = []
    total_palavras = 0

    for unidade in unidades:
        palavras = unidade.split()
        n_palavras = len(palavras)

        # Se adicionar essa unidade ultrapassar o limite, salva o chunk atual
        if total_palavras + n_palavras > chunk_size and chunk:
            chunks.append("; ".join(chunk))
            chunk = []
            total_palavras = 0

        chunk.append(unidade)
        total_palavras += n_palavras

    # Salva o último chunk
    if chunk:
        chunks.append("; ".join(chunk))

    return chunks
