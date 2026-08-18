# src/ingestor.py

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import config

def load_and_ingest():
    print("Loading PDFs...")
    pdfLoader = DirectoryLoader(
        config.PDF_PATH_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    pdfDocuments = pdfLoader.load()
    print(f"Loaded {len(pdfDocuments)} pages across all PDFs")
    docxLoader = DirectoryLoader(
        config.PDF_PATH_DIR, 
        glob='**/*.docx', 
        loader_cls=UnstructuredWordDocumentLoader
    )
    docxDocuments = docxLoader.load()
    print(f"Loaded {len(docxDocuments)} pages across all docx's")

    allDocuments = pdfDocuments + docxDocuments
    
    print("Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(allDocuments)
    print(f"Created {len(chunks)} chunks")

    print("Embedding and storing in ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.CHROMA_DB_PATH
    )
    print(f"Done. {len(chunks)} chunks stored in ChromaDB")
    return vectorstore


if __name__ == "__main__":
    load_and_ingest()