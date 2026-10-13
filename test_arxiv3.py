import requests

url = "http://export.arxiv.org/api/query?id_list=2407.10793"
try:
    resp = requests.get(url, timeout=15)
    print(resp.status_code)
    if resp.status_code == 200:
        print(resp.text[:500])
except Exception as e:
    print(e)
