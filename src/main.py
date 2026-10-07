from rag import RAGPipeline


def main():
    print("=" * 70)
    print("Technical Knowledge RAG Assistant")
    print("=" * 70)
    print("Ask questions about the documents.")
    print("Type 'exit' to quit.\n")

    rag = RAGPipeline()

    while True:
        question = input("Question: ").strip()

        if question.lower() == "exit":
            print("\nGoodbye!")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        answer, results = rag.generate_answer(
            question,
            top_k=3,
        )

        print(f"\nAnswer: {answer}\n")

        print("Retrieved sources:")
        for result in results:
            print(
                f"- {result['source']} "
                f"(similarity: {result['score']:.4f})"
            )

        print()


if __name__ == "__main__":
    main()