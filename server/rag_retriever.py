# truth_seeker_env/server/rag_retriever.py
"""
RAG retriever using Qdrant as the vector database.

Two modes:
  1. Qdrant mode (production) — used when QDRANT_URL is set.
     Embeds queries with sentence-transformers/all-MiniLM-L6-v2 locally.
  2. Local fallback (dev/test) — keyword-overlap search over task's
     internal_docs. Scores are keyword-overlap ratios, NOT cosine
     similarities. Only suitable for unit tests / offline dev runs.
"""

import re
import logging

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Optional imports — only needed in production Qdrant mode
# ──────────────────────────────────────────────────────────────────────────────
try:
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False
    logger.warning(
        "qdrant-client or sentence-transformers not installed. "
        "QdrantRAGRetriever will use local keyword fallback."
    )


class QdrantRAGRetriever:
    """
    Retrieves relevant document chunks for a given query.

    Production: Qdrant vector search + local sentence-transformer embeddings.
    Dev/test:   LocalFallbackRetriever (keyword overlap over internal_docs).
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str = "",
        collection_name: str = "truth-seeker",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self._use_qdrant = bool(url) and _QDRANT_AVAILABLE

        if self._use_qdrant:
            try:
                self._client = QdrantClient(url=url, api_key=api_key)
                self._collection_name = collection_name
                self._embedder = SentenceTransformer(embedding_model)
                logger.info(
                    f"QdrantRAGRetriever connected to '{url}' collection='{collection_name}' "
                    f"using model '{embedding_model}'"
                )
            except Exception as e:
                logger.error(
                    f"Qdrant connection failed (url='{url}'): {e}. "
                    "Falling back to local retriever."
                )
                self._use_qdrant = False

        if not self._use_qdrant:
            logger.info("QdrantRAGRetriever: using LOCAL FALLBACK (dev/test mode).")
            self._fallback = LocalFallbackRetriever()
        else:
            self._fallback = None

        self._episode_docs: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_episode_docs(self, internal_docs: dict):
        """
        Called by environment.reset() each episode.
        Wires per-episode docs into the local fallback retriever.
        Ignored in Qdrant mode (docs are already in the collection).
        """
        self._episode_docs = internal_docs
        if self._fallback is not None:
            self._fallback.load(internal_docs)

    def search(self, query: str, top_k: int = 3) -> str:
        """
        Search for relevant document chunks.
        Returns formatted string of chunks, or a "no results" message.
        """
        if not query or not query.strip():
            return "No results: empty query."

        if self._use_qdrant:
            return self._qdrant_search(query, top_k)
        return self._fallback.search(query, top_k)

    def has_relevant_results(self, query: str, threshold: float = 0.7) -> bool:
        """
        True if any chunk scores above threshold.
        threshold=0.7 is calibrated for Qdrant cosine similarity.
        Fallback uses threshold=0.3 (keyword overlap ratios are not comparable).
        """
        if self._use_qdrant:
            return self._qdrant_has_relevant(query, threshold)
        return self._fallback.has_relevant(query, threshold=0.3)

    # ------------------------------------------------------------------
    # Qdrant implementation
    # ------------------------------------------------------------------

    def _qdrant_search(self, query: str, top_k: int) -> str:
        try:
            embedding = self._embedder.encode(query).tolist()
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=embedding,
                limit=top_k
            )
            chunks = [
                match.payload.get("text", "")
                for match in response.points
                if match.payload and "text" in match.payload
            ]
            if not chunks:
                return "No relevant documents found."
            return "\n\n---\n\n".join(chunks)
        except Exception as e:
            logger.error(f"Qdrant search error: {e}")
            return f"Search error: {e}"

    def _qdrant_has_relevant(self, query: str, threshold: float) -> bool:
        try:
            embedding = self._embedder.encode(query).tolist()
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=embedding,
                limit=1
            )
            return bool(response.points) and response.points[0].score >= threshold
        except Exception as e:
            logger.error(f"Qdrant has_relevant error: {e}")
            return False


# ──────────────────────────────────────────────────────────────────────────────
# Local Fallback — keyword overlap over in-memory docs
# ──────────────────────────────────────────────────────────────────────────────

class LocalFallbackRetriever:
    """
    Keyword-overlap retriever for dev/test without Qdrant.
    Scores are Jaccard ratios (not cosine similarities) — only for local use.
    """

    def __init__(self):
        self._docs: dict = {}  # doc_id -> content string

    def load(self, internal_docs: dict):
        self._docs = internal_docs or {}

    def search(self, query: str, top_k: int = 3) -> str:
        if not self._docs:
            return "No documents available (local fallback — no internal_docs loaded)."

        query_tokens = set(_tokenize(query.lower()))
        if not query_tokens:
            return "No results: empty query tokens."

        scored = []
        for doc_id, content in self._docs.items():
            doc_tokens = set(_tokenize(content.lower()))
            union = query_tokens | doc_tokens
            overlap = len(query_tokens & doc_tokens)
            score = overlap / len(union) if union else 0.0
            if score > 0:
                scored.append((score, doc_id, content))

        if not scored:
            return "No relevant documents found."

        scored.sort(key=lambda x: x[0], reverse=True)
        chunks = [content for _, _, content in scored[:top_k]]
        return "\n\n---\n\n".join(chunks)

    def has_relevant(self, query: str, threshold: float = 0.3) -> bool:
        if not self._docs:
            return False
        query_tokens = set(_tokenize(query.lower()))
        for content in self._docs.values():
            doc_tokens = set(_tokenize(content.lower()))
            union = query_tokens | doc_tokens
            score = len(query_tokens & doc_tokens) / len(union) if union else 0.0
            if score >= threshold:
                return True
        return False


def _tokenize(text: str) -> list:
    return re.findall(r'\b\w+\b', text)
