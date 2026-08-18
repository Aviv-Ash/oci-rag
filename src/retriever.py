from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import config

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=config.CHROMA_DB_PATH,
        embedding_function=embeddings
    )
    return vectorstore


def retrieve(question: str, vectorstore) -> list:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string")
    
    results = vectorstore.similarity_search(question, k=config.TOP_K_RESULTS)
    return results