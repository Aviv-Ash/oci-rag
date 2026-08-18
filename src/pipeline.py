from src.guardrail import is_oci_related
from src.retriever import load_vectorstore, retrieve
from src.generator import generate

vectorstore = load_vectorstore()

def run(question: str) -> dict:
    if not is_oci_related(question):
        return {
            "answer": "I'm only able to answer questions about Oracle Cloud Infrastructure (OCI). Please ask an OCI-related question.",
            "sources": []
        }

    chunks = retrieve(question, vectorstore)
    result = generate(question, chunks)
    return result