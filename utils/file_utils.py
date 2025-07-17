import os, glob, csv, json
import pandas as pd
from utils.embedding_utils import format_doc_e5
from config.settings import DATA_FOLDER

def read_jsonl_from_folder(folder_path):
    jsonl_files = glob.glob(os.path.join(folder_path, "*.jsonl"))
    documents = []
    for file in jsonl_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        texto = obj.get("texto", "")
                        metadata = {k: v for k, v in obj.items() if k != "texto"}
                        if texto:
                            documents.append({
                                "text": texto,
                                "metadata": metadata
                            })
        except Exception as e:
            print(f"❌ Erro ao ler {file}: {e}")
            continue
    return documents

def process_documents(source, files, options):
    if source == "Pasta padrão":
        return read_jsonl_from_folder(DATA_FOLDER)
    else:
        documents = []
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        texto = obj.get("texto", "")
                        metadata = {k: v for k, v in obj.items() if k != "texto"}
                        if texto:
                            if options['embedding_model'] == "bge-base-pt":
                                texto = format_doc_e5(texto)
                            documents.append({
                                "text": texto,
                                "metadata": metadata
                            })
        return documents