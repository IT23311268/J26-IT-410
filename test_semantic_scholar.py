import requests
import re

def normalize_text(text):
    if not text: return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def query_semantic_scholar(title):
    """Search Semantic Scholar for a paper by title."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": title,
        "limit": 3,
        "fields": "title,authors,year,externalIds",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            for item in data:
                t = item.get("title", "")
                authors = [a.get("name", "") for a in item.get("authors", [])]
                year = str(item.get("year", ""))
                ext = item.get("externalIds", {})
                doi = ext.get("DOI")
                arxiv = ext.get("ArXiv")
                print(f"  Title: {t}")
                print(f"  Authors: {authors}")
                print(f"  Year: {year}")
                print(f"  ArXiv: {arxiv}")
                print(f"  DOI: {doi}")
                print()
            return data
    except Exception as e:
        print(f"Error: {e}")
    return None

# Test with GraphEval
print("=== Test 1: GraphEval ===")
query_semantic_scholar("GraphEval Knowledge-Graph LLM hallucination evaluation framework")

print("=== Test 2: SelfCheckGPT ===")
query_semantic_scholar("SelfCheckGPT Zero-Resource Black-Box Hallucination Detection")
