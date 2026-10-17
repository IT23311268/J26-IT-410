"""
Service layer for Literature Review operations.
Stores literature reviews in memory for PP1 (will be replaced with persistent storage later).
"""

import uuid
from datetime import datetime, timezone
from typing import Optional


class LiteratureService:
    """
    Manages literature review storage and retrieval.
    Uses an in-memory store (class-level dict) for PP1.
    """

    _store: dict = {}

    def save_literature_review(self, text: str) -> dict:
        """Save a literature review and return its metadata."""
        text = text.strip()
        if len(text) < 50:
            raise ValueError(
                "Literature review is too short. Please provide at least 50 characters."
            )

        review_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        record = {
            "id": review_id,
            "literature_review": text,
            "submitted_at": now,
        }
        LiteratureService._store[review_id] = record

        return {
            "id": review_id,
            "message": "Literature review submitted successfully.",
            "literature_review_preview": text[:200] + ("..." if len(text) > 200 else ""),
            "character_count": len(text),
            "submitted_at": now,
        }

    def get_literature_review(self, review_id: str) -> Optional[dict]:
        """Retrieve a literature review by its ID."""
        return LiteratureService._store.get(review_id)
