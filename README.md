# OCI Assistant — RAG-Powered Documentation Chatbot

A local RAG (Retrieval-Augmented Generation) pipeline designed to answer
Oracle Cloud Infrastructure (OCI) questions grounded strictly in official
OCI documentation — with no hallucination and no off-topic responses.

---

## Architecture

User Question
↓
Guardrail (two-stage classifier)
↓ NO ↓ YES
Polite refusal Retriever (ChromaDB similarity search)
↓
Generator (LLM + context + citations)
↓
Answer + Source Citations

Six components, each with one job:

- **config.py** — single source of truth for all settings
- **ingestor.py** — loads PDFs, chunks them, embeds and stores in ChromaDB. Runs once, independent of the query path.
- **retriever.py** — loads the vector store, embeds the user question, returns top K most relevant chunks
- **guardrail.py** — dedicated LLM classifier that decides if the question is OCI-related before the pipeline runs
- **generator.py** — builds the prompt from retrieved context, calls the LLM, returns answer + citations
- **pipeline.py** — orchestrates the above components in order

---

## Tech Stack

| Tool                                     | Purpose                       | Why                                                    |
| ---------------------------------------- | ----------------------------- | ------------------------------------------------------ |
| Ollama + llama3.2:3b                     | Local LLM                     | Free, runs on GPU, no API key                          |
| sentence-transformers (all-MiniLM-L6-v2) | Embeddings                    | Fast, local, matches ChromaDB's cosine similarity      |
| ChromaDB                                 | Vector store                  | Simple, local, persists to disk                        |
| LangChain                                | Document loading and chunking | Handles PDF parsing and RecursiveCharacterTextSplitter |
| pypdf                                    | PDF text extraction           | Lightweight, no external dependencies                  |
| RAGAS                                    | RAG assessment                | gives you an estimate about your RAG                   |

---

## Design Decisions

**Two-stage guardrail over a system prompt**
A system prompt relies on the same LLM that generates the answer to also
police the input — a single point of failure that can be manipulated. A
dedicated classifier is a separate gate with one job. It's harder to bypass,
easier to debug, and provides a clean place to log refused queries.

**Ingestion separated from query pipeline**
Ingestion is a one-time cost that only re-runs when documents change.
Separating it means the vector store loads once at startup and is reused
across all queries. Swapping ChromaDB for pgvector in production only
touches one file.

**RecursiveCharacterTextSplitter over naive chunking**
Splits on natural boundaries (paragraphs → sentences → words) before
resorting to character splits. Produces semantically coherent chunks,
which improves both retrieval accuracy and generation quality.

**Same embedding model at ingestion and query time**
Embeddings must live in the same vector space to be comparable. Using
different models at ingestion vs. query time would produce incompatible
vectors and break retrieval entirely.

**K=5 retrieved chunks**
Balances context richness against prompt noise. Too few chunks risk missing
relevant content. Too many dilute the context with loosely related chunks,
degrading answer quality. K=5 is the baseline — the eval harness in eval/
is designed to tune this empirically.

---

## Setup

```bash
# Clone and enter the project
git clone <repo-url>
cd oci-rag

# Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
source venv/bin/activate       # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Pull the LLM
ollama pull llama3.2:3b

# Add your OCI PDFs to data/
# Then run ingestion (only needed once, or when docs change)
python -m src.ingestor

# Start the assistant
python -m main
```

---

## What I'd Add With More Time

- **Web UI / Telegram bot** — replace the terminal loop with a chat interface accessible from anywhere
- **Conversation memory** — currently stateless; add a message history so the assistant remembers context across turns
- **Streaming responses** — stream tokens as they generate instead of waiting for the full answer
- **Expand the document library** — ingest the full OCI documentation suite, not just one guide
- **Add tool calling** — Like let the LLM answer questions he cant right now. (i.e. using web search)

---

## Project Structure

oci-rag/
├── data/ # PDFs & docx and ChromaDB vector store
├── src/
│ ├── ingestor.py # Ingestion pipeline
│ ├── retriever.py # Vector search
│ ├── guardrail.py # Topic classifier
│ ├── generator.py # LLM generation
│ └── pipeline.py # Orchestration
├── eval/
│ └── test_questions.json
│ └── evaluate.py
├── main.py # Entry point
├── config.py # All settings
└── README.md
