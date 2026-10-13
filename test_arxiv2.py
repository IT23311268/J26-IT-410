import requests
import urllib.parse
import xml.etree.ElementTree as ET

def query_arxiv_search(title: str):
    q = urllib.parse.quote(title)
    url = f"http://export.arxiv.org/api/query?search_query=ti:%22{q}%22&max_results=3"
    print(url)
    resp = requests.get(url, timeout=10)
    if resp.status_code == 200:
        root = ET.fromstring(resp.text)
        entries = root.findall("{http://www.w3.org/2005/Atom}entry")
        print(f"Found {len(entries)} entries")
        for entry in entries:
            t = entry.find("{http://www.w3.org/2005/Atom}title").text.replace("\n", " ").strip()
            print("TITLE:", t)
            
            published = entry.find("{http://www.w3.org/2005/Atom}published").text
            print("YEAR:", published[:4] if published else "")
            
            id_el = entry.find("{http://www.w3.org/2005/Atom}id")
            arxiv_id = id_el.text.split('/')[-1] if id_el is not None else None
            print("ARXIV ID:", arxiv_id)
            
            authors = [author.find("{http://www.w3.org/2005/Atom}name").text for author in entry.findall("{http://www.w3.org/2005/Atom}author")]
            print("AUTHORS:", authors)
    else:
        print("Status", resp.status_code)

query_arxiv_search("GraphEval")
