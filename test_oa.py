import requests

def verify_openalex(ref_text: str):
    url = "https://api.openalex.org/works"
    params = {
        "search": ref_text,
        "per-page": 1,
        "mailto": "admin@example.com"
    }
    
    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        items = data.get("results", [])
        if not items:
            return "Not found"
        return {"title": items[0].get("title"), "doi": items[0].get("doi")}
    else:
        return f"Error {resp.status_code}"

ref1 = "Hamadi, R.: Large language models meet computer vision: A brief survey (2023) https://doi.org/10.48550/arXiv.2311.16673 arXiv:2311.16673 [cs.CV]"
print("OpenAlex match:")
print(verify_openalex(ref1))
