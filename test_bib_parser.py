import re
from typing import List, Dict, Optional

def parse_bibliography_entries(bib_text: str) -> List[Dict]:
    if re.search(r"^\s*\[\d+\]\s", bib_text, re.MULTILINE):
        entries = []
        matches = list(re.finditer(r"^\s*\[(\d+)\]\s", bib_text, re.MULTILINE))
        for i, match in enumerate(matches):
            ref_num = match.group(1)
            start_pos = match.end()
            end_pos = matches[i+1].start() if i + 1 < len(matches) else len(bib_text)
            ref_content = bib_text[start_pos:end_pos].replace('\n', ' ').strip()
            if ref_content:
                entries.append({
                    "ref_num": ref_num,
                    "ref_text": ref_content
                })
        return entries
        
    if re.search(r"^\s*\d+\.\s", bib_text, re.MULTILINE):
        entries = []
        matches = list(re.finditer(r"^\s*(\d+)\.\s", bib_text, re.MULTILINE))
        for i, match in enumerate(matches):
            ref_num = match.group(1)
            start_pos = match.end()
            end_pos = matches[i+1].start() if i + 1 < len(matches) else len(bib_text)
            ref_content = bib_text[start_pos:end_pos].replace('\n', ' ').strip()
            if ref_content:
                entries.append({
                    "ref_num": ref_num,
                    "ref_text": ref_content
                })
        return entries
        
    refs = re.split(r"\n\s*\n", bib_text)
    entries = []
    for r in refs:
        r = r.replace('\n', ' ').strip()
        if len(r) > 20:
            entries.append({
                "ref_num": None,
                "ref_text": r
            })
    return entries

bib_text = """
[1] Manakul, P., Liusie, A., Gales, M.J.F.: SelfCheckGPT. (2023).
[5] Smith, J., Jones, A.: Another Paper. (2024).
"""
print(parse_bibliography_entries(bib_text))
