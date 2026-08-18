import ollama
import config

CLASSIFIER_PROMPT = """
You are a strict topic classifier. 
Your only job is to decide if a question is related to Oracle Cloud Infrastructure (OCI).

Answer with a single word only: YES or NO.

Examples:
Q: How do I create a compute instance in OCI? → YES
Q: What is the weather today? → NO
Q: How do I configure a VCN? → YES
Q: Who is the president of the USA? → NO
Q: What storage options does OCI offer? → YES
Q: How do I make pasta? → NO
"""

def is_oci_related(question: str) -> bool:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string")

    response = ollama.chat(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": CLASSIFIER_PROMPT},
            {"role": "user", "content": f"Q: {question}"}
        ]
    )

    answer = response["message"]["content"].strip().upper()
    return answer.startswith("YES")