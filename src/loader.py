from pathlib import Path


def load_documents(directory: str | Path) -> list[dict]:
    """
    Load all non-empty .txt documents from a directory.

    Returns:
        A list of dictionaries containing:
        - text: document content
        - source: original filename
    """
    directory = Path(directory)

    documents = []

    for file_path in sorted(directory.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        documents.append(
            {
                "text": text,
                "source": file_path.name,
            }
        )

    return documents


if __name__ == "__main__":
    # Find the project root automatically.
    project_root = Path(__file__).resolve().parent.parent

    documents_path = project_root / "data" / "documents"

    documents = load_documents(documents_path)

    print(f"Loaded {len(documents)} documents.\n")

    for document in documents:
        print(f"Source: {document['source']}")
        print(f"Characters: {len(document['text'])}")
        print("-" * 50)