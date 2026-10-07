from pathlib import Path

import chromadb

from chunker import chunk_documents
from embeddings import EmbeddingModel
from loader import load_documents


class ChromaVectorStore:
    """
    Stores document chunks and their embeddings in ChromaDB.
    """

    def __init__(
        self,
        directory: str | Path,
        collection_name: str = "technical_knowledge",
    ):
        self.directory = str(directory)
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(
            path=self.directory
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            configuration={
                "hnsw": {
                    "space": "cosine"
                }
            },
        )

    def build(
        self,
        embeddings,
        chunks: list[dict],
    ) -> None:
        """
        Add embeddings and chunk metadata to ChromaDB.
        """

        if len(embeddings) != len(chunks):
            raise ValueError(
                "The number of embeddings must match "
                "the number of chunks."
            )

        # Remove old data so repeated runs start cleanly.
        if self.collection.count() > 0:
            self.client.delete_collection(
                name=self.collection_name
            )

            self.collection = self.client.create_collection(
                name=self.collection_name,
                configuration={
                    "hnsw": {
                        "space": "cosine"
                    }
                },
            )

        ids = [
            f"{chunk['source']}_{chunk['chunk_id']}"
            for chunk in chunks
        ]

        documents = [
            chunk["text"]
            for chunk in chunks
        ]

        metadatas = [
            {
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
            }
            for chunk in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Search ChromaDB for the most relevant chunks.
        """

        top_k = min(
            top_k,
            self.collection.count(),
        )

        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        retrieved = []

        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]

            # For cosine distance, smaller distance means
            # greater similarity.
            similarity = 1.0 - distance

            retrieved.append(
                {
                    "text": results["documents"][0][i],
                    "source": results["metadatas"][0][i]["source"],
                    "chunk_id": results["metadatas"][0][i]["chunk_id"],
                    "score": float(similarity),
                }
            )

        return retrieved


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    documents_path = project_root / "data" / "documents"
    chroma_path = project_root / "data" / "chroma_store"

    # 1. Load documents
    documents = load_documents(documents_path)

    # 2. Create chunks
    chunks = chunk_documents(documents)

    # 3. Create embeddings
    embedding_model = EmbeddingModel()

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.encode(texts)

    # 4. Build ChromaDB store
    vector_store = ChromaVectorStore(chroma_path)
    vector_store.build(embeddings, chunks)

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Chroma records: {vector_store.collection.count()}")

    # 5. Test retrieval
    query = "What is machine learning?"

    query_embedding = embedding_model.encode([query])

    results = vector_store.search(
        query_embedding,
        top_k=3,
    )

    print(f"\nQuery: {query}\n")

    for result in results:
        print(f"Source: {result['source']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Similarity: {result['score']:.4f}")
        print(f"Text: {result['text']}")
        print("-" * 60)