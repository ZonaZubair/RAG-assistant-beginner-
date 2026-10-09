from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from chunker import chunk_documents
from embeddings import EmbeddingModel
from loader import load_documents
from retriever import Retriever
from vector_store import FAISSVectorStore


class RAGPipeline:
    """
    Complete Retrieval-Augmented Generation pipeline.

    If a saved FAISS index exists, it is loaded.
    Otherwise, the index is built automatically from
    the documents in data/documents/.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
    ):
        project_root = Path(__file__).resolve().parent.parent

        documents_path = project_root / "data" / "documents"
        vector_store_path = project_root / "data" / "vector_store"

        print("Loading embedding model...")
        self.embedding_model = EmbeddingModel()

        index_path = vector_store_path / "index.faiss"
        metadata_path = vector_store_path / "metadata.json"

        if index_path.exists() and metadata_path.exists():
            print("Loading FAISS vector store...")

            self.vector_store = FAISSVectorStore.load(
                vector_store_path
            )

        else:
            print("FAISS index not found. Building vector store...")

            documents = load_documents(documents_path)
            chunks = chunk_documents(documents)

            texts = [chunk["text"] for chunk in chunks]

            embeddings = self.embedding_model.encode(texts)

            self.vector_store = FAISSVectorStore()

            self.vector_store.build(
                embeddings,
                chunks,
            )

            # Save the generated index when possible.
            try:
                self.vector_store.save(vector_store_path)
                print("FAISS vector store saved.")
            except Exception as error:
                print(f"Could not save FAISS store: {error}")

        self.retriever = Retriever(
            self.embedding_model,
            self.vector_store,
        )

        print("Loading language model...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name
        )

        print("RAG pipeline ready.\n")

    def generate_answer(
        self,
        query: str,
        top_k: int = 3,
    ) -> tuple[str, list[dict]]:
        """
        Retrieve relevant context and generate an answer.
        """

        results = self.retriever.retrieve(
            query,
            top_k=top_k,
        )

        context_parts = []

        for result in results:
            context_parts.append(
                f"Source: {result['source']}\n"
                f"{result['text']}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{query}

Answer:
""".strip()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
        )

        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        return answer, results


if __name__ == "__main__":
    rag = RAGPipeline()

    questions = [
        "What is machine learning?",
        "What is NLP used for?",
        "What is RAG?",
        "What libraries are commonly used in Python?",
    ]

    for question in questions:
        print("=" * 70)
        print(f"Question: {question}")

        answer, results = rag.generate_answer(
            question,
            top_k=3,
        )

        print(f"\nAnswer: {answer}")

        print("\nRetrieved sources:")
        for result in results:
            print(
                f"- {result['source']} "
                f"(similarity: {result['score']:.4f})"
            )

        print()