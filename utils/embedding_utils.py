from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.preprocessing import normalize
import numpy as np

def get_embedder(model_name):
    if model_name == "multilingual_e5_large":
        return HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    return HuggingFaceEmbeddings(model_name="BAAI/bge-base-pt")

def chunk_text(text, chunk_size):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def format_doc_e5(text): return "passage: " + text
def format_query_e5(text): return "query: " + text
