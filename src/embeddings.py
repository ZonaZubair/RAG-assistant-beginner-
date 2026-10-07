from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Creates vector embeddings for text.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Convert a list of texts into normalized float32 vectors.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(embeddings, dtype=np.float32)


if __name__ == "__main__":
    from loader import load_documents
    from chunker import chunk_documents

    project_root = Path(__file__).resolve().parent.parent
    documents_path = project_root / "data" / "documents"

    documents = load_documents(documents_path)
    chunks = chunk_documents(documents)

    texts = [chunk["text"] for chunk in chunks]

    embedding_model = EmbeddingModel()
    embeddings = embedding_model.encode(texts)

    print(f"Chunks: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Vector dimensions: {embeddings.shape[1]}")