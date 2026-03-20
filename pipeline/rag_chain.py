"""RAG chain: combines Retriever + Generator into a single ask() call.

Usage:
    from pipeline.rag_chain import RAGChain

    chain = RAGChain(model="groq")
    answer = chain.ask("What are the symptoms of diabetes?")
"""

from typing import Any

from config.settings import RETRIEVAL_TOP_K
from llm.base import BaseLLM
from llm.bedrock_client import BedrockLLM
from llm.groq_client import GroqLLM
from pipeline.generator import Generator
from pipeline.retriever import Retriever
from vectorstore.embedder import Embedder
from vectorstore.store import VectorStore


def _create_llm(model: str) -> BaseLLM:
    """Instantiate the requested LLM backend.

    Args:
        model: "groq" or "bedrock".

    Returns:
        A BaseLLM instance.
    """
    if model == "groq":
        return GroqLLM()
    elif model == "bedrock":
        return BedrockLLM()
    else:
        raise ValueError(f"Unknown model: {model!r}. Use 'groq' or 'bedrock'.")


class RAGChain:
    """End-to-end RAG pipeline: question → retrieval → generation → answer."""

    def __init__(
        self,
        model: str = "groq",
        top_k: int = RETRIEVAL_TOP_K,
    ) -> None:
        """Initialize the full RAG chain.

        Args:
            model: LLM backend — "groq" or "bedrock".
            top_k: Number of chunks to retrieve per query.
        """
        self.top_k = top_k

        # Shared embedder and store for the retriever
        embedder = Embedder()
        store = VectorStore()
        self.retriever = Retriever(embedder=embedder, store=store)

        llm = _create_llm(model)
        self.generator = Generator(llm=llm)

        self.model_name = model
        print(f"RAG chain ready (model={model}, top_k={top_k})")

    def ask(
        self,
        question: str,
        demographic_context: str | None = None,
    ) -> dict[str, Any]:
        """Ask a health question and get a cited answer.

        Args:
            question: Natural language health question.
            demographic_context: Optional demographic info string
                to inject into the generation prompt.

        Returns:
            Dict with keys:
                - answer: The generated response text.
                - sources: List of source metadata dicts.
                - model: Which LLM backend was used.
        """
        # Retrieve
        chunks = self.retriever.retrieve(question, top_k=self.top_k)

        # Generate
        answer = self.generator.generate(
            question, chunks, demographic_context=demographic_context,
        )

        # Collect source metadata for the caller
        sources = [
            {
                "pmid": c["metadata"]["pmid"],
                "title": c["metadata"]["title"],
                "journal": c["metadata"]["journal"],
                "year": c["metadata"]["year"],
                "url": c["metadata"]["url"],
                "authors": c["metadata"]["authors"],
            }
            for c in chunks
        ]

        # Raw context texts for evaluation (RAGAS needs these)
        contexts = [c["document"] for c in chunks]

        return {
            "answer": answer,
            "contexts": contexts,
            "sources": sources,
            "model": self.model_name,
        }
