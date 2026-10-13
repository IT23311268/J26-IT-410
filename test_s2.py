import requests

def verify_s2(ref_text: str):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": ref_text,
        "limit": 1,
        "fields": "title,authors,year"
    }
    
    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        items = data.get("data", [])
        if not items:
            return "Not found"
        return items[0]
    else:
        return f"Error {resp.status_code}"

ref1 = "Hamadi, R.: Large language models meet computer vision: A brief survey (2023) https://doi.org/10.48550/arXiv.2311.16673 arXiv:2311.16673 [cs.CV]"
print("S2 match:")
print(verify_s2(ref1))
