import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from duckduckgo_search import DDGS  # 改用原生導入，避開 LangChain Bug
import ollama

# --- 設定區 ---
DATA_PATH = "data/"
DB_PATH = "vector_db"
MODEL_NAME = "kenneth85/llama-3-taiwan"

def initialize_vector_db():
    """初始化向量資料庫"""
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"✅ 已建立 {DATA_PATH} 資料夾。")
    
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    
    if not documents:
        print("⚠️ 提示：data 資料夾內沒有 PDF，僅能使用閒聊與網路搜尋。")
        return None

    print("📖 正在索引 PDF 文件...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=MODEL_NAME),
        persist_directory=DB_PATH
    )
    print(f"✅ 索引完成！")
    return vector_db

def web_search(query):
    """原生 DuckDuckGo 搜尋函數，解決 ddgs 模組遺失問題"""
    try:
        with DDGS() as ddgs:
            # 搜尋並取前 3 筆結果的摘要
            results = [r['body'] for r in ddgs.text(query, max_results=3)]
            return "\n".join(results)
    except Exception as e:
        return f"搜尋功能暫時失靈 (原因: {e})"

def ask_ai(query, vector_db):
    """智慧決策大腦"""
    pdf_context = ""
    if vector_db:
        results = vector_db.similarity_search(query, k=2)
        pdf_context = "\n\n".join([doc.page_content for doc in results])
    
    # 觸發網路搜尋的關鍵字
    web_context = ""
    keywords = ["最新", "今天", "天氣", "公告", "新聞", "誰是", "推薦"]
    
    if len(pdf_context.strip()) < 50 or any(word in query for word in keywords):
        print("🔍 助教麻吉正在上網打聽中...")
        web_context = web_search(query)

    system_prompt = f"""
    你現在是虎科大專業且幽默的「助教麻吉」。
    
    【知識來源】：
    - 內部文件 (PDF)：{pdf_context if pdf_context else "無相關資料"}
    - 網路情報：{web_context if web_context else "無相關資料"}
    
    【規則】：
    1. 全程使用「台灣繁體中文」回答。
    2. 語氣親切，多用「安啦、沒問題、喔」。
    3. 若資訊來自網路，請註明「我幫你上網打聽了一下...」。
    """
    
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': query}
        ],
        options={'temperature': 0.7}
    )
    return response['message']['content']

def main():
    db = initialize_vector_db()
    
    print("\n" + "="*40)
    print(f"🚀 虎科大 AI 萬事通 ({MODEL_NAME}) 已啟動！")
    print("我可以讀 PDF 也能幫你上網查資料。")
    print("="*40)

    while True:
        user_input = input("\n你：")
        if user_input.lower() == 'exit':
            break
        
        if not user_input.strip():
            continue

        print("AI 思考中...")
        ans = ask_ai(user_input, db)
        print(f"\n助教麻吉：\n{ans}")

if __name__ == "__main__":
    main()