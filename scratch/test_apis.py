import requests

title = "Language Models are Unsupervised Multitask Learners"

# Test direct OpenAI URL
r_oa = requests.head("https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf", timeout=5)
print("OpenAI CDN head status:", r_oa.status_code)

# Test Wikipedia API
r_wiki = requests.get("https://en.wikipedia.org/w/api.php", params={
    "action": "query", "list": "search", "srsearch": f'"{title}"', "format": "json"
}, headers={"User-Agent": "HallucinationDetector/1.0"}, timeout=5)
print("Wiki status:", r_wiki.status_code)
if r_wiki.status_code == 200:
    for s in r_wiki.json().get('query', {}).get('search', [])[:3]:
        print("  Wiki hit:", s.get('title'), s.get('snippet')[:100])

# Test Wikidata API
r_wd = requests.get("https://www.wikidata.org/w/api.php", params={
    "action": "wbsearchentities", "search": title, "language": "en", "format": "json"
}, headers={"User-Agent": "HallucinationDetector/1.0"}, timeout=5)
print("Wikidata status:", r_wd.status_code)
if r_wd.status_code == 200:
    for item in r_wd.json().get('search', [])[:3]:
        print("  WD hit:", item.get('label'), item.get('description'))
