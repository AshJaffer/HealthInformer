"""Retriever: query string → top-k relevant chunks from ChromaDB.

Loads the PubMedBERT embedder and ChromaDB store, embeds the user
query, and returns the most relevant chunks with metadata.

Usage:
    from pipeline.retriever import Retriever

    retriever = Retriever()
    results = retriever.retrieve("What causes diabetes?", top_k=5)
"""

from typing import Any

from config.settings import RETRIEVAL_TOP_K
from vectorstore.embedder import Embedder
from vectorstore.store import VectorStore


class Retriever:
    """Embeds a query and retrieves relevant chunks from the vector store."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
    ) -> None:
        """Initialize with an embedder and vector store.

        Args:
            embedder: Pre-loaded Embedder instance. Created if None.
            store: Pre-loaded VectorStore instance. Created if None.
        """
        self.embedder = embedder or Embedder()
        self.store = store or VectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> list[dict[str, Any]]:
        """Retrieve the top-k most relevant chunks for a query.

        Args:
            query: Natural language question from the user.
            top_k: Number of chunks to retrieve.

        Returns:
            List of result dicts, each with keys:
                id, document, metadata, distance.
        """
        query_embedding = self.embedder.embed(query)
        results = self.store.query(query_embedding, top_k=top_k)
        return results


if __name__ == "__main__":
    retriever = Retriever()
    query = "What are the warning signs of type 2 diabetes?"
    results = retriever.retrieve(query, top_k=5)
    print(f"Query: {query!r}\n")
    for i, r in enumerate(results, 1):
        m = r["metadata"]
        print(f"[{i}] dist={r['distance']:.4f} | {m['title'][:70]}")
