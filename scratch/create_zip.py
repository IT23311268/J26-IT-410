import zipfile
import os

files_to_pack = [
    "backend/services/reference_verifier.py",
    "backend/models/schemas.py",
    "backend/test_verifier.py",
    "frontend/src/App.js",
    "frontend/src/components/ReferenceVerification.js",
]

output_zip = "citation_verification_changes.zip"

with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for rel_path in files_to_pack:
        if os.path.exists(rel_path):
            zf.write(rel_path, arcname=rel_path)
            print(f"Added: {rel_path}")
        else:
            print(f"Warning: file not found: {rel_path}")

print(f"\nSUCCESS: Created archive '{output_zip}' with {len(zf.namelist())} files.")
