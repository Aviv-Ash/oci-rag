import ollama
import config

SYSTEM_PROMPT = """
You are an expert Oracle Cloud Infrastructure (OCI) assistant.
Your job is to directly and concisely answer the user's question using ONLY the context provided.

Rules:
1. Read the question carefully and answer it directly — do not add unrequested information
2. Use ONLY information from the provided context — never use outside knowledge
3. If the answer is not in the context, say exactly: "I don't have enough information in my documents to answer that."
4. After your answer, cite the source document and page number
5. Keep your answer focused — if the question asks how to do X, explain how to do X and nothing else
"""

def format_context(chunks: list) -> tuple[str, list]:
    context_parts = []
    sources = []

    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "Unknown")
        page = chunk.metadata.get("page", "Unknown")
        context_parts.append(f"[{i+1}] (Source: {source}, Page: {page})\n{chunk.page_content}")
        sources.append({"index": i+1, "source": source, "page": page})

    context_str = "\n\n".join(context_parts)
    return context_str, sources


def generate(question: str, chunks: list) -> dict:
    context_str, sources = format_context(chunks)

    prompt = f"""Context:
{context_str}

Question: {question}
"""

    response = ollama.chat(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )

    return {
        "answer": response["message"]["content"],
        "sources": sources
    }