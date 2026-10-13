import fitz
import re
import requests
from pathlib import Path

def extract_references_from_pdf(pdf_path: str):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
        
    lines = full_text.split("\n")
    ref_start_idx = -1
    for i in range(len(lines)-1, max(-1, len(lines)-1000), -1):
        line = lines[i].strip().lower()
        if line in ["references", "bibliography", "references.", "bibliography."]:
            ref_start_idx = i
            break
            
    if ref_start_idx == -1:
        for i in range(len(lines)-1, max(-1, len(lines)-1000), -1):
            line = lines[i].strip().lower()
            if re.match(r"^\d*\.?\s*(references|bibliography)$", line):
                ref_start_idx = i
                break
                
    if ref_start_idx == -1:
        print("Could not find references section.")
        return []
        
    ref_text = "\n".join(lines[ref_start_idx+1:])
    
    # Heuristic parsing
    if re.search(r"^\s*\[1\]\s", ref_text, re.MULTILINE) or re.search(r"^\s*1\.\s", ref_text, re.MULTILINE):
        refs = re.split(r"^\s*\[\d+\]\s|^\s*\d+\.\s", ref_text, flags=re.MULTILINE)
        refs = [r.strip().replace("\n", " ") for r in refs if r.strip()]
    else:
        # Paragraph based
        refs = re.split(r"\n\s*\n", ref_text)
        refs = [r.strip().replace("\n", " ") for r in refs if r.strip() and len(r.strip()) > 20]
        
    return refs

def verify_reference(ref_text: str):
    url = "https://api.crossref.org/works"
    params = {
        "query.bibliographic": ref_text,
        "rows": 1,
        "select": "title,author,DOI"
    }
    headers = {"User-Agent": "HallucinationDetector/1.0 (mailto:admin@example.com)"}
    
    resp = requests.get(url, params=params, headers=headers, timeout=10)
    if resp.status_code == 200:
        items = resp.json().get("message", {}).get("items", [])
        if not items:
            return {"status": "FAKE", "reason": "No match found in Crossref."}
            
        top = items[0]
        title = top.get("title", [""])[0] if top.get("title") else ""
        
        # Token overlap
        title_tokens = set(re.findall(r'\w+', title.lower()))
        ref_tokens = set(re.findall(r'\w+', ref_text.lower()))
        
        if not title_tokens:
             return {"status": "UNKNOWN", "reason": "No title in matched record."}
             
        overlap = len(title_tokens.intersection(ref_tokens)) / len(title_tokens)
        
        if overlap >= 0.7:
            return {"status": "REAL", "match": title, "score": overlap}
        elif overlap >= 0.4:
            return {"status": "UNCERTAIN", "match": title, "score": overlap}
        else:
            return {"status": "FAKE", "match": title, "score": overlap}
    else:
        return {"status": "ERROR", "reason": f"API error {resp.status_code}"}

pdf_file = list(Path("uploads").glob("*.pdf"))[0]
print(f"Extracting references from {pdf_file}...")
refs = extract_references_from_pdf(str(pdf_file))
print(f"Found {len(refs)} references.")

for i, ref in enumerate(refs[:3]):
    print(f"\n--- Ref {i+1} ---")
    print(f"Text: {ref}")
    res = verify_reference(ref)
    print(f"Verification: {res}")
