import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))

from rag import RAGPipeline


st.set_page_config(
    page_title="Technical Knowledge RAG Assistant",
    page_icon="📚",
    layout="centered",
)

st.title("📚 Technical Knowledge RAG Assistant")

st.write(
    "Ask questions about Python, Machine Learning, NLP, "
    "and RAG using the information in the document collection."
)

st.caption(
    "Answers are generated from retrieved document context. "
    "Check the sources to verify the information."
)


@st.cache_resource
def load_rag():
    return RAGPipeline()


with st.spinner("Loading the RAG system..."):
    rag = load_rag()


# Using a form allows Enter to submit the question.
question = st.chat_input(
    "Ask a question about Python, Machine Learning, NLP, or RAG..."
)

if question:
    try:
        with st.spinner("Searching documents and generating an answer..."):
            answer, results = rag.generate_answer(
                question.strip(),
                top_k=3,
            )

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Retrieved Sources")

        if not results:
            st.info("No relevant document chunks were retrieved.")

        for result in results:
            title = (
                f"{result['source']} | "
                f"Similarity: {result['score']:.4f}"
            )

            with st.expander(title):
                st.write(result["text"])

    except Exception as error:
        st.error(
            "Something went wrong while processing your question."
        )
        st.exception(error)