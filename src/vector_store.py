import json
from pathlib import Path

import faiss
import numpy as np


class FAISSVectorStore:
    """
    Stores embeddings in a FAISS index and keeps
    the corresponding chunk metadata.
    """

    def __init__(self):
        self.index = None
        self.metadata = []

    def build(
        self,
        embeddings: np.ndarray,
        chunks: list[dict],
    ) -> None:
        """
        Build a FAISS index from embeddings.
        """
        if len(embeddings) == 0:
            raise ValueError("No embeddings were provided.")

        if len(embeddings) != len(chunks):
            raise ValueError(
                "The number of embeddings must match the number of chunks."
            )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype(np.float32))

        self.metadata = chunks

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Search for the most similar chunks.
        """
        if self.index is None:
            raise ValueError("Vector store has not been built yet.")

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        top_k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            result = self.metadata[index].copy()
            result["score"] = float(score)

            results.append(result)

        return results

    def save(self, directory: str | Path) -> None:
        """
        Save the FAISS index and metadata to disk.
        """
        if self.index is None:
            raise ValueError("Nothing to save. Build the vector store first.")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(directory / "index.faiss"),
        )

        with open(
            directory / "metadata.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(self.metadata, file, indent=2)

    @classmethod
    def load(cls, directory: str | Path):
        """
        Load a previously saved FAISS vector store.
        """
        directory = Path(directory)

        store = cls()

        store.index = faiss.read_index(
            str(directory / "index.faiss")
        )

        with open(
            directory / "metadata.json",
            "r",
            encoding="utf-8",
        ) as file:
            store.metadata = json.load(file)

        return store


if __name__ == "__main__":
    from loader import load_documents
    from chunker import chunk_documents
    from embeddings import EmbeddingModel

    project_root = Path(__file__).resolve().parent.parent
    documents_path = project_root / "data" / "documents"
    vector_store_path = project_root / "data" / "vector_store"

    # 1. Load documents
    documents = load_documents(documents_path)

    # 2. Create chunks
    chunks = chunk_documents(documents)

    # 3. Create embeddings
    embedding_model = EmbeddingModel()

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.encode(texts)

    # 4. Build vector store
    vector_store = FAISSVectorStore()
    vector_store.build(embeddings, chunks)

    # 5. Save vector store
    vector_store.save(vector_store_path)

    # 6. Test a search
    query = "What is machine learning?"
    query_embedding = embedding_model.encode([query])

    results = vector_store.search(query_embedding, top_k=3)

    print(f"\nQuery: {query}\n")

    for result in results:
        print(f"Source: {result['source']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Similarity: {result['score']:.4f}")
        print(f"Text: {result['text']}")
        print("-" * 60)