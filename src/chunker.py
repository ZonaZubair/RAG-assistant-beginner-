from pathlib import Path
import re


def chunk_text(
    text: str,
    chunk_size: int = 80,
    overlap: int = 15,
) -> list[str]:
    """
    Strategy 1: Fixed-size word chunking with overlap.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))

        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return chunks


def sentence_chunk_text(
    text: str,
    sentences_per_chunk: int = 3,
) -> list[str]:
    """
    Strategy 2: Sentence-based chunking.

    Groups a fixed number of complete sentences together.
    """
    if sentences_per_chunk <= 0:
        raise ValueError("sentences_per_chunk must be greater than 0")

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    chunks = []

    for start in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(
            sentences[start:start + sentences_per_chunk]
        ).strip()

        if chunk:
            chunks.append(chunk)

    return chunks


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 80,
    overlap: int = 15,
) -> list[dict]:
    """
    Create chunks using Strategy 1.
    """
    all_chunks = []

    for document in documents:
        chunks = chunk_text(
            document["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_id, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "text": chunk,
                    "source": document["source"],
                    "chunk_id": chunk_id,
                }
            )

    return all_chunks


if __name__ == "__main__":
    from loader import load_documents

    project_root = Path(__file__).resolve().parent.parent
    documents_path = project_root / "data" / "documents"

    documents = load_documents(documents_path)

    # Strategy 1
    fixed_chunks = chunk_documents(documents)

    print("Strategy 1: Fixed-size word chunking")
    print(f"Documents loaded: {len(documents)}")
    print(f"Total chunks: {len(fixed_chunks)}")
    print()

    # Strategy 2
    print("Strategy 2: Sentence-based chunking")

    sentence_chunk_count = 0

    for document in documents:
        chunks = sentence_chunk_text(document["text"])
        sentence_chunk_count += len(chunks)

        print(
            f"{document['source']}: "
            f"{len(chunks)} chunks"
        )

    print(f"Total sentence-based chunks: {sentence_chunk_count}")