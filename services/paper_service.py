"""
Service layer for reference paper (PDF) operations.
Handles file saving with safe filenames and metadata tracking.
"""

import os
import re
import uuid
from datetime import datetime, timezone
from typing import List
from fastapi import UploadFile


class PaperService:
    """Manages reference paper upload, storage, and listing."""

    def __init__(self, uploads_dir: str):
        self.uploads_dir = uploads_dir
        os.makedirs(self.uploads_dir, exist_ok=True)

    def _make_safe_filename(self, original: str) -> str:
        """Generate a safe, unique filename from the original."""
        # Remove path separators and keep only safe characters
        name = os.path.basename(original)
        name_without_ext = os.path.splitext(name)[0]
        # Replace unsafe characters with underscores
        safe_name = re.sub(r"[^\w\-.\ ]", "_", name_without_ext)
        safe_name = safe_name.strip().replace(" ", "_")
        # Add a short unique suffix to avoid collisions
        short_id = uuid.uuid4().hex[:8]
        return f"{safe_name}_{short_id}.pdf"

    async def save_papers(self, files: List[UploadFile]) -> dict:
        """Save uploaded PDF files and return metadata."""
        papers_metadata = []

        for file in files:
            safe_name = self._make_safe_filename(file.filename)
            file_path = os.path.join(self.uploads_dir, safe_name)

            # Read and write file contents
            contents = await file.read()
            with open(file_path, "wb") as f:
                f.write(contents)

            papers_metadata.append({
                "original_filename": file.filename,
                "saved_filename": safe_name,
                "file_size_bytes": len(contents),
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            })

        return {
            "message": f"Successfully uploaded {len(papers_metadata)} paper(s).",
            "total_uploaded": len(papers_metadata),
            "papers": papers_metadata,
        }

    def list_papers(self) -> list:
        """List all uploaded papers with their metadata."""
        papers = []
        if not os.path.exists(self.uploads_dir):
            return papers

        for filename in sorted(os.listdir(self.uploads_dir)):
            if filename.lower().endswith(".pdf"):
                filepath = os.path.join(self.uploads_dir, filename)
                papers.append({
                    "saved_filename": filename,
                    "file_size_bytes": os.path.getsize(filepath),
                })
        return papers
