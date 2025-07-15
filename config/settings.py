import os
from datetime import datetime

DATA_FOLDER = "fonte_de_dados/dados_abertos"
INDEX_DIR = "faiss_indices"
EMBEDDING_CACHE_DIR = "embedding_cache"

os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)

DEFAULT_INDEX_VERSION = datetime.now().strftime("%Y%m%d_%H%M%S")
