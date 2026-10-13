import requests
import re
import time
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def normalize_text(text):
    if not text: return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def calculate_similarity(title, ref_text):
    t_norm = normalize_text(title)
    r_norm = normalize_text(ref_text)
    if not t_norm or not r_norm: return 0.0
    if t_norm in r_norm:
        return 1.0 if len(t_norm.split()) > 2 else 0.9
    t_tokens = set(t_norm.split())
    r_tokens = set(r_norm.split())
    if not t_tokens: return 0.0
    overlap = len(t_tokens.intersection(r_tokens)) / len(t_tokens)
    return overlap

# Test OpenAlex for GraphEval
ref_text = 'H. Sansford, N. Richardson, H. P. Maretic, and J. N. Saada, "GraphEval: A Knowledge-Graph based LLM hallucination evaluation framework," 2024.'
approx_title = "GraphEval: A Knowledge-Graph based LLM hallucination evaluation framework"

print("=== OpenAlex Test ===")
url = "https://api.openalex.org/works"
params = {"search": approx_title, "per-page": 3, "mailto": "admin@example.com"}
resp = requests.get(url, params=params, timeout=10)
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    items = resp.json().get("results", [])
    print(f"Results: {len(items)}")
    for item in items[:3]:
        title = item.get("title") or ""
        authors = [a.get("author", {}).get("display_name", "") for a in item.get("authorships", [])]
        year = str(item.get("publication_year")) if item.get("publication_year") else ""
        doi_url = item.get("doi")
        doi = doi_url.replace("https://doi.org/", "") if doi_url else None
        sim = calculate_similarity(title, ref_text)
        print(f"  Title: {title}")
        print(f"  Authors: {authors}")
        print(f"  Year: {year}")
        print(f"  DOI: {doi}")
        print(f"  Similarity: {sim:.2f}")
        
        # Check for arXiv in locations
        for loc in item.get("locations", []):
            source = loc.get("source", {})
            if source and "arxiv" in (source.get("display_name", "")).lower():
                print(f"  arXiv source found!")
            landing = loc.get("landing_page_url", "") or ""
            if "arxiv.org" in landing:
                arxiv_id = landing.split("/")[-1]
                print(f"  arXiv ID from URL: {arxiv_id}")
        print()

print("\n=== Crossref Test ===")
url2 = "https://api.crossref.org/works"
params2 = {"query.bibliographic": ref_text, "rows": 2, "select": "title,author,DOI,issued"}
headers = {"User-Agent": "HallucinationDetector/1.0 (mailto:admin@example.com)"}
resp2 = requests.get(url2, params=params2, headers=headers, timeout=5)
print(f"Status: {resp2.status_code}")
if resp2.status_code == 200:
    items2 = resp2.json().get("message", {}).get("items", [])
    for item in items2[:2]:
        title = item.get("title", [""])[0] if item.get("title") else ""
        sim = calculate_similarity(title, ref_text)
        print(f"  Title: {title}")
        print(f"  Similarity: {sim:.2f}")
