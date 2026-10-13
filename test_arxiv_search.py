import requests
import xml.etree.ElementTree as ET
import urllib.parse

title = "GraphEval"
url = f"http://export.arxiv.org/api/query?search_query=all:%22{urllib.parse.quote(title)}%22&max_results=3"
print(url)
resp = requests.get(url)
print(resp.status_code)
root = ET.fromstring(resp.text)
for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
    t = entry.find("{http://www.w3.org/2005/Atom}title").text
    print("Title:", t.replace('\n', ' '))
    auths = [author.find("{http://www.w3.org/2005/Atom}name").text for author in entry.findall("{http://www.w3.org/2005/Atom}author")]
    print("Authors:", auths)
    id_el = entry.find("{http://www.w3.org/2005/Atom}id")
    print("ID:", id_el.text.split('/')[-1])
