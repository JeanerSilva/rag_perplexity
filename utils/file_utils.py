import os, glob, csv
import pandas as pd
from utils.embedding_utils import format_doc_e5, chunk_text
from config.settings import DATA_FOLDER

def detect_delimiter(file_path):
    with open(file_path, 'r', encoding='latin1') as f:
        sample = f.read(2048)
        sniffer = csv.Sniffer()
        return sniffer.sniff(sample).delimiter

def read_csvs_from_folder(folder_path):
    texts = []
    for file in glob.glob(os.path.join(folder_path, "*.csv")):
        try:
            delimiter = detect_delimiter(file)
            df = pd.read_csv(file, encoding="latin1", sep=delimiter, engine='python', on_bad_lines='skip')
            for _, row in df.iterrows():
                row_text = "; ".join([f"{col}: {row[col]}" for col in df.columns if pd.notnull(row[col])])
                texts.append(row_text)
        except Exception as e:
            print(f"Erro ao ler {file}: {e}")
    return texts

def process_documents(source, files, options):
    if source == "Pasta padrão":
        return read_csvs_from_folder(DATA_FOLDER)
    else:
        documents = []
        for file in files:
            df = pd.read_csv(file)
            for _, row in df.iterrows():
                texto = f"Programa {row['CODIGO']}: {row['OBJETIVO']}\nMetas: {row['METAS']}"
                chunks = chunk_text(texto, options['chunk_size'])
                if options['embedding_model'] == "multilingual_e5_large":
                    chunks = [format_doc_e5(chunk) for chunk in chunks]
                documents.extend(chunks)
        return documents
