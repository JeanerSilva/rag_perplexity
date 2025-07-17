import os, glob, json
from config.settings import DATA_FOLDER
from utils.embedding_utils import chunk_text, format_doc_e5


def read_jsonl_files(jsonl_files, chunk_size, embedding_model):
    documents = []

    for file in jsonl_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue

                    obj = json.loads(line)
                    texto = obj.get("texto", "")
                    metadados = {k: v for k, v in obj.items() if k != "texto"}

                    if texto:
                        chunks = chunk_text(texto, chunk_size)
                        for i, chunk in enumerate(chunks):
                            if embedding_model == "bge-base-pt":
                                chunk = format_doc_e5(chunk)

                            documents.append({
                                "text": chunk,
                                "metadata": {
                                    **metadados,
                                    "chunk_id": i + 1
                                }
                            })
        except Exception as e:
            print(f"❌ Erro ao ler {file}: {e}")
            continue

    return documents


def read_jsonl_from_folder(folder_path, chunk_size, embedding_model):
    jsonl_files = glob.glob(os.path.join(folder_path, "*.jsonl"))
    return read_jsonl_files(jsonl_files, chunk_size, embedding_model)


def process_documents(source, files, options):
    chunk_size = options["chunk_size"]
    embedding_model = options["embedding_model"]

    if source == "Pasta padrão":
        return read_jsonl_from_folder(DATA_FOLDER, chunk_size, embedding_model)
    else:
        jsonl_files = [file.name for file in files]

        # Salva temporariamente os arquivos carregados manualmente (opcional)
        for file in files:
            with open(file.name, "wb") as out:
                out.write(file.getbuffer())

        return read_jsonl_files(jsonl_files, chunk_size, embedding_model)
