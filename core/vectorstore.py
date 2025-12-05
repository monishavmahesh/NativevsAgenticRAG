import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Ollama embeddings model
EMBEDDING_MODEL_NAME = "nomic-embed-text"

def build_vector_store(subject: str, docs, persist_root: str = "./db"):
    persist_dir = os.path.join(persist_root, subject)
    os.makedirs(persist_dir, exist_ok=True)
    
    # ✅ Use OllamaEmbeddings
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    
    # ✅ IMPROVED: Larger chunks for Math to keep problems complete
    if subject.lower() == "math":
        chunk_size = 800      # Doubled for math problems
        chunk_overlap = 200   # More overlap to preserve context
    else:
        chunk_size = 400
        chunk_overlap = 100
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]  # ✅ Better splitting
    )
    
    chunks = splitter.split_documents(docs)
    
    print(f"[VectorStore] Created {len(chunks)} chunks for {subject}")
    
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata or {}
        meta.setdefault("subject", subject)
        meta.setdefault("chunk_id", i)  # ✅ Add chunk ID for debugging
        chunk.metadata = meta
    
    # Build FAISS index
    vectordb = FAISS.from_documents(chunks, embeddings)
    
    # Persist FAISS index
    index_path = os.path.join(persist_dir, "faiss_index")
    vectordb.save_local(index_path)
    
    print(f"[VectorStore] Saved index to {index_path}")
    
    return vectordb

def load_vector_store(subject: str, persist_root: str = "./db"):
    persist_dir = os.path.join(persist_root, subject)
    index_path = os.path.join(persist_dir, "faiss_index")
    
    if not os.path.exists(index_path):
        return None
    
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    vectordb = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    return vectordb
