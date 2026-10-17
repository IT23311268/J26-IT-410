import re
from typing import List, Dict, Optional

UNICODE_SURNAME = r"[^\W\d_](?:[^\W\d_]|['\-])*"

def resolve_citation_to_reference(citation_text: str, refs: List[Dict]) -> Optional[Dict]:
    if not citation_text:
        return None
        
    # Check for numeric citations like [5] or [5, 6] (we'll just take the first number for simplicity or try to match)
    # Actually the citation_text might be [5]
    num_match = re.search(r"\[\s*(\d+)", citation_text)
    if num_match:
        ref_num = num_match.group(1)
        for r in refs:
            if r.get("ref_num") == ref_num:
                return r
                
    # Otherwise, it's author-year like (Smith et al., 2024)
    author_match = re.search(UNICODE_SURNAME, citation_text, flags=re.UNICODE)
    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", citation_text)
    
    author = author_match.group().lower() if author_match else None
    year = year_match.group(1) if year_match else None
    
    if not author:
        return None
        
    for r in refs:
        ref_text_lower = r["ref_text"].lower()
        if author in ref_text_lower:
            if year and year in ref_text_lower:
                return r
            elif not year:
                return r
                
    return None

def _match_reference_to_paper(ref: Dict, metadata_dict: Dict[str, dict]) -> Optional[str]:
    # Extract details from ref
    ref_text = ref["ref_text"]
    doi_match = re.search(r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b", ref_text, re.I)
    doi = doi_match.group(1).rstrip('.') if doi_match else None
    
    arxiv_match = re.search(r"arxiv:\s*(\d{4}\.\d{4,5}(?:v\d+)?)", ref_text, re.I)
    if not arxiv_match:
        arxiv_match = re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)", ref_text, re.I)
    arxiv = arxiv_match.group(1) if arxiv_match else None
    
    author_match = re.search(UNICODE_SURNAME, ref_text, flags=re.UNICODE)
    author = author_match.group().lower() if author_match else None
    
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", ref_text)
    year = int(years[-1]) if years else None
    
    # Priority 1: DOI match
    if doi:
        for filename, metadata in metadata_dict.items():
            if metadata.get("doi") == doi:
                return filename
                
    # Priority 2: arXiv match
    if arxiv:
        for filename, metadata in metadata_dict.items():
            if metadata.get("arxiv") == arxiv:
                return filename
                
    if not author:
        return None
        
    # Priority 3: Author + Year
    for filename, metadata in metadata_dict.items():
        author_surnames = [s.lower() for s in metadata.get("author_surnames", [])]
        metadata_year = metadata.get("year")
        metadata_years = metadata.get("years", [])
        
        author_matches = any((author in s or s in author) for s in author_surnames)
        year_matches = (year is None or metadata_year == year or year in metadata_years)
        
        if author_matches and year_matches:
            return filename
            
    # Priority 4: Author only
    for filename, metadata in metadata_dict.items():
        author_surnames = [s.lower() for s in metadata.get("author_surnames", [])]
        author_matches = any((author in s or s in author) for s in author_surnames)
        if author_matches:
            return filename
            
    return None

refs = [
    {"ref_num": "5", "ref_text": "Manakul, P., Liusie, A.: SelfCheckGPT (2023) arxiv:2303.08896"},
    {"ref_num": "6", "ref_text": "Smith, J. (2024)"}
]

metadata_dict = {
    "selfcheckgpt.pdf": {
        "author_surnames": ["Manakul", "Liusie"],
        "year": 2023,
        "arxiv": "2303.08896"
    }
}

ref = resolve_citation_to_reference("[5]", refs)
print("Resolved:", ref)
if ref:
    matched_paper = _match_reference_to_paper(ref, metadata_dict)
    print("Matched Paper:", matched_paper)
