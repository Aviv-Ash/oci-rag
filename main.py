from src.pipeline import run

def main():
    print("OCI Assistant ready. Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            print("Goodbye.")
            break
        if not question:
            continue

        result = run(question)
        print(f"\nAssistant: {result['answer']}")
        if result["sources"]:
            print("\nSources:")
            for s in result["sources"]:
                print(f"  [{s['index']}] {s['source']} — Page {s['page']}")
        print()

if __name__ == "__main__":
    main()