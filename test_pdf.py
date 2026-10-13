import sys
sys.path.insert(0, "backend")
from services.reference_verifier import extract_references_from_pdf
refs = extract_references_from_pdf("backend/uploads/2023.eacl-main.199_ad404a46.pdf")
print("Extracted", len(refs))
