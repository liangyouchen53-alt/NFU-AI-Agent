import os

# 模型名稱與 README.md 指令對應
DATA_DIR = "data/"
DB_PATH = "faiss_index" 
MODEL_NAME = "kenneth85/llama-3-taiwan" 
EMBED_MODEL = "nomic-embed-text" 

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)