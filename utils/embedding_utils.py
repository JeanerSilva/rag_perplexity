from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.preprocessing import normalize
import numpy as np

def get_embedder(model_name):
    if model_name == "multilingual_e5_large":
        return HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    return HuggingFaceEmbeddings(model_name="BAAI/bge-base-pt")

def chunk_text(text, chunk_size):
    """
    Divide o texto em chunks respeitando os delimitadores `;` como unidades lógicas,
    agrupando essas unidades até atingir o tamanho máximo de palavras (chunk_size).
    Nunca quebra uma unidade no meio.
    """
    unidades = [u.strip() for u in text.replace("\n", " ").split(";") if u.strip()]
    chunks = []
    chunk = []
    total_palavras = 0

    for unidade in unidades:
        palavras = unidade.split()
        n_palavras = len(palavras)

        if total_palavras + n_palavras > chunk_size and chunk:
            chunks.append("; ".join(chunk))
            chunk = []
            total_palavras = 0

        chunk.append(unidade)
        total_palavras += n_palavras

    if chunk:
        chunks.append("; ".join(chunk))

    return chunks


def format_doc_e5(text): return "passage: " + text
def format_query_e5(text): return "query: " + text
