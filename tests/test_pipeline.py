import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))

from loader import load_documents
from chunker import chunk_documents


class TestRAGPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.documents_path = (
            PROJECT_ROOT / "data" / "documents"
        )

    def test_documents_are_loaded(self):
        documents = load_documents(self.documents_path)

        self.assertEqual(len(documents), 4)

        for document in documents:
            self.assertIn("text", document)
            self.assertIn("source", document)

    def test_chunks_are_created(self):
        documents = load_documents(self.documents_path)

        chunks = chunk_documents(documents)

        self.assertGreater(len(chunks), 0)

        for chunk in chunks:
            self.assertIn("text", chunk)
            self.assertIn("source", chunk)
            self.assertIn("chunk_id", chunk)


if __name__ == "__main__":
    unittest.main()