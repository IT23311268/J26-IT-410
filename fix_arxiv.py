import re

with open("backend/services/reference_verifier.py", "r", encoding="utf-8") as f:
    content = f.read()

arxiv_search_func = """
def query_arxiv_search(title: str) -> dict:
    import urllib.parse
    import xml.etree.ElementTree as ET
    try:
        # ArXiv API struggles with exact quoted titles sometimes if they include punctuation.
        # We'll use a simple all:query search.
        clean_title = re.sub(r'[^\\w\\s]', ' ', title).strip()
        words = clean_title.split()[:8]
        if not words: return None
        q = " ".join(words)
        url = f"http://export.arxiv.org/api/query?search_query=all:%22{urllib.parse.quote(q)}%22&max_results=2"
        
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            root = ET.fromstring(resp.text)
            for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
                t = entry.find("{http://www.w3.org/2005/Atom}title").text.replace("\\n", " ").strip()
                authors = [author.find("{http://www.w3.org/2005/Atom}name").text for author in entry.findall("{http://www.w3.org/2005/Atom}author")]
                published = entry.find("{http://www.w3.org/2005/Atom}published").text
                year = published[:4] if published else ""
                
                id_el = entry.find("{http://www.w3.org/2005/Atom}id")
                arxiv_id = id_el.text.split('/')[-1] if id_el is not None else None
                
                return {"source": "arXiv", "title": t, "authors": authors, "year": year, "doi": None, "arxiv": arxiv_id, "raw": {"title": [t], "author": [{"family": a} for a in authors]}}
    except: pass
    return None
"""

# Insert it before query_crossref_biblio
if "def query_arxiv_search" not in content:
    content = content.replace("def query_crossref_biblio", arxiv_search_func + "\ndef query_crossref_biblio")
    
old_logic = """    # Priority 5: Crossref
    cand = query_crossref_biblio(ref_text)
    if cand: candidates.append(cand)
    
    # Priority 6: Fallback OpenAlex"""

new_logic = """    # Priority 5: Crossref
    cand = query_crossref_biblio(ref_text)
    if cand: candidates.append(cand)
    
    # Priority 5.5: arXiv fallback search
    if not candidates or all(calculate_similarity(c["title"], ref_text) < 0.8 for c in candidates):
        cand_arxiv = query_arxiv_search(parsed["approx_title"])
        if cand_arxiv: candidates.append(cand_arxiv)
    
    # Priority 6: Fallback OpenAlex"""

content = content.replace(old_logic, new_logic)

with open("backend/services/reference_verifier.py", "w", encoding="utf-8", newline="\n") as f:
    f.write(content)
