from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from embeddings import EmbeddingModel
from vector_store import FAISSVectorStore
from retriever import Retriever


class RAGPipeline:
    """
    Complete Retrieval-Augmented Generation pipeline.

    1. Retrieve relevant chunks.
    2. Build a prompt using those chunks.
    3. Generate an answer using a language model.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
    ):
        print("Loading embedding model...")
        self.embedding_model = EmbeddingModel()

        project_root = Path(__file__).resolve().parent.parent
        vector_store_path = project_root / "data" / "vector_store"

        print("Loading FAISS vector store...")
        self.vector_store = FAISSVectorStore.load(
            vector_store_path
        )

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