import requests
import json
import time

# Wait a moment if the server needs to start
# Actually, I am not starting a server, I can just call the service function directly for testing.
from services.reference_verifier import verify_document_references
import os

pdf_file = os.path.join("uploads", "v1_covered_83f58e01-73ab-4238-a32f-e65c70c5f73d_a0f8b952.pdf")
print(f"Testing document: {pdf_file}")
result = verify_document_references(pdf_file)
print(json.dumps(result, indent=2))
