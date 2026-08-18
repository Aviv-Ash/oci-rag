import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.guardrail import is_oci_related
from src.retriever import load_vectorstore, retrieve
from src.generator import generate
import config

def run_evaluation():
    print("Loading eval questions...")
    with open("eval/test_questions.json", "r") as f:
        test_data = json.load(f)

    vectorstore = load_vectorstore()

    ragas_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    print("\nRunning tests...\n")
    for test in test_data["tests"]:
        question = test["question"]
        category = test["category"]
        print(f"[{category.upper()}] {question}")

        if category == "rejected":
            rejected = not is_oci_related(question)
            status = "✓ PASSED" if rejected else "✗ FAILED"
            print(f"  Guardrail: {status}\n")
            continue

        if category == "unanswerable":
            chunks = retrieve(question, vectorstore)
            result = generate(question, chunks)
            knows_limit = "don't have enough information" in result["answer"].lower()
            status = "✓ PASSED" if knows_limit else "✗ FAILED"
            print(f"  Knows its limits: {status}\n")
            continue

        if category == "answerable":
            chunks = retrieve(question, vectorstore)
            result = generate(question, chunks)

            ragas_data["question"].append(question)
            ragas_data["answer"].append(result["answer"])
            ragas_data["contexts"].append([c.page_content for c in chunks])
            ragas_data["ground_truth"].append(test.get("ground_truth", ""))

            print(f"  Answer: {result['answer'][:100]}...\n")

    print("\nRunning RAGAS scoring on answerable questions...")
    llm = LangchainLLMWrapper(ChatOllama(model=config.LLM_MODEL))
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    )

    dataset = Dataset.from_dict(ragas_data)
    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings
    )

    print("\n--- RAGAS Results ---")
    print(scores)

if __name__ == "__main__":
    run_evaluation()