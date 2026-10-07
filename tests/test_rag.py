import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))

from rag import RAGPipeline


class TestRAGQuestions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rag = RAGPipeline()

    def test_machine_learning_question(self):
        answer, results = self.rag.generate_answer(
            "What is machine learning?",
            top_k=3,
        )

        self.assertTrue(answer.strip())
        self.assertTrue(len(results) > 0)
        self.assertEqual(
            results[0]["source"],
            "machine_learning.txt",
        )

    def test_nlp_question(self):
        answer, results = self.rag.generate_answer(
            "What is NLP?",
            top_k=3,
        )

        self.assertTrue(answer.strip())
        self.assertEqual(
            results[0]["source"],
            "natural_language_processing.txt",
        )

    def test_rag_question(self):
        answer, results = self.rag.generate_answer(
            "What is RAG?",
            top_k=3,
        )

        self.assertTrue(answer.strip())
        self.assertEqual(
            results[0]["source"],
            "rag.txt",
        )

    def test_python_question(self):
        answer, results = self.rag.generate_answer(
            "What libraries are commonly used in Python?",
            top_k=3,
        )

        self.assertTrue(answer.strip())
        self.assertEqual(
            results[0]["source"],
            "python.txt",
        )


if __name__ == "__main__":
    unittest.main()