import re

with open("backend/services/reference_verifier.py", "r", encoding="utf-8") as f:
    content = f.read()

old_func = """def extract_approx_title(ref_text: str) -> str:
    text = re.sub(r"https?://\S+", "", ref_text)
    text = re.sub(r"arxiv:\s*\d+\.\d+", "", text, flags=re.I)

    # Priority 1: Extract title inside quotes (most reliable)
    quoted = re.search(r'"|`(.*?)"|''', text)
    if quoted:
        title = quoted.group(1).strip()
        if len(title.split()) >= 3:
            return title

    parts = re.split(r"[\.:]\s+", text)"""

new_func = """def extract_approx_title(ref_text: str) -> str:
    text = re.sub(r"https?://\S+", "", ref_text)
    text = re.sub(r"arxiv:\s*\d+\.\d+", "", text, flags=re.I)
    
    # Priority 1: Extract title inside quotes (most reliable)
    quoted = re.search(r'["\u201c\u201d\u2018\u2019](.+?)["\u201c\u201d\u2018\u2019]', text)
    if quoted:
        title = quoted.group(1).strip()
        if len(title.split()) >= 3:
            return title
            
    parts = re.split(r"[\.:]\s+", text)"""

content = content.replace(old_func, new_func)

with open("backend/services/reference_verifier.py", "w", encoding="utf-8", newline="\n") as f:
    f.write(content)
