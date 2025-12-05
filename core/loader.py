from langchain_community.document_loaders import PyPDFLoader

def load_documents(file_path):
    """Load PDF and return list of documents."""
    loader = PyPDFLoader(file_path)
    return loader.load()
