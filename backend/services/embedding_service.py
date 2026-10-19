"""
Embedding & FAISS Vector Search Service.

Uses the sentence-transformers library with the all-MiniLM-L6-v2 model
to create embeddings, builds a FAISS index, and performs similarity search.
Fully local — no API keys required.
"""

import numpy as np
from typing import List, Optional

import faiss
from sentence_transformers import SentenceTransformer

# ── Singleton model loader ───────────────────────────────────────────────────

_model: Optional[SentenceTransformer] = None
MODEL_NAME = "all-MiniLM-L6-v2"


def _get_model() -> SentenceTransformer:
    """Lazy-load the sentence-transformer model (downloaded once)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# ── Embedding Generation ────────────────────────────────────────────────────

def generate_embeddings(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Generate embeddings for a list of texts.

    Returns:
        A numpy array of shape (len(texts), embedding_dim).
    """
    import torch
    model = _get_model()
    with torch.inference_mode():
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Enables cosine similarity via dot product
        )
    return embeddings.astype("float32")



# ── FAISS Index ──────────────────────────────────────────────────────────────

class FAISSIndex:
    """
    Wraps a FAISS index for building and searching over text chunks.
    Uses Inner Product (equivalent to cosine similarity when vectors are normalized).
    """

    def __init__(self):
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[dict] = []  # Parallel list of chunk metadata

    def build(self, chunks: List[dict]) -> int:
        """
        Build the FAISS index from text chunks.

        Args:
            chunks: List of {"chunk_id", "paper", "page", "text"} dicts.

        Returns:
            Number of chunks indexed.
        """
        if not chunks:
            raise ValueError("No chunks provided to build the FAISS index.")

        self.chunks = chunks
        texts = [c["text"] for c in chunks]

        embeddings = generate_embeddings(texts)
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        return self.index.ntotal

    def search(
        self,
        query_text: str,
        top_k: int = 3,
        paper_filter: Optional[str] = None,
    ) -> List[dict]:
        """
        Search the index for chunks most similar to the query.

        Args:
            query_text: The claim text to search for.
            top_k: Number of top results to return.
            paper_filter: If provided, prioritize results from this paper.

        Returns:
            List of evidence dicts:
                {
                    "chunk_id": "...",
                    "paper": "...",
                    "page": 5,
                    "chunk_text": "...",
                    "similarity_score": 0.82,
                    "match_type": "citation_match" | "similarity_based"
                }
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        query_embedding = generate_embeddings([query_text])

        # Fetch more candidates so we can filter/prioritize
        fetch_k = min(top_k * 5, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, fetch_k)

        scores = scores[0]
        indices = indices[0]

        # Build result candidates
        candidates = []
        for score, idx in zip(scores, indices):
            if idx < 0:  # FAISS returns -1 for missing entries
                continue
            chunk = self.chunks[idx]
            candidates.append({
                "chunk_id": chunk["chunk_id"],
                "paper": chunk["paper"],
                "page": chunk["page"],
                "chunk_text": chunk["text"][:500],  # Truncate for response size
                "similarity_score": round(float(score), 4),
                "match_type": "similarity_based",
            })

        # If a paper filter is provided, strictly restrict matches to that paper
        if paper_filter:
            matched = []
            for c in candidates:
                if paper_filter.lower() in c["paper"].lower():
                    c["match_type"] = "citation_match"
                    matched.append(c)
            candidates = matched

        return candidates[:top_k]
