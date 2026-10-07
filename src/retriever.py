from pathlib import Path

from embeddings import EmbeddingModel
from vector_store import FAISSVectorStore


class Retriever:
    """
    Retrieves the most relevant document chunks
    for a user's question.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: FAISSVectorStore,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Convert the query into an embedding and
        retrieve the most similar chunks.
        """
        query_embedding = self.embedding_model.encode([query])

        return self.vector_store.search(
            query_embedding,
            top_k=top_k,
        )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    vector_store_path = project_root / "data" / "vector_store"

    embedding_model = EmbeddingModel()
    vector_store = FAISSVectorStore.load(vector_store_path)

    retriever = Retriever(
        embedding_model,
        vector_store,
    )

    query = "What are the different types of machine learning?"

    results = retriever.retrieve(query, top_k=3)

    print(f"Query: {query}\n")

    for result in results:
        print(f"Source: {result['source']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Similarity: {result['score']:.4f}")
        print(f"Text: {result['text']}")
        print("-" * 60)