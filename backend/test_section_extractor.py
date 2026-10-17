import re
from services.reference_verifier import parse_bibliography_text

def extract_literature_review_section(text: str) -> str:
    lines = text.split('\n')
    lr_started = False
    lr_lines = []
    
    lr_headings = [
        r"^\s*(?:\d+\.?\s*)?literature\s+review\s*$",
        r"^\s*(?:\d+\.?\s*)?related\s+work\s*$",
        r"^\s*(?:\d+\.?\s*)?background\s*$",
        r"^\s*(?:\d+\.?\s*)?background\s+and\s+related\s+work\s*$"
    ]
    
    end_headings = [
        r"^\s*(?:\d+\.?\s*)?methodology\s*$",
        r"^\s*(?:\d+\.?\s*)?materials\s+and\s+methods\s*$",
        r"^\s*(?:\d+\.?\s*)?experiment[s]?\s*$",
        r"^\s*(?:\d+\.?\s*)?experimental\s+setup\s*$",
        r"^\s*(?:\d+\.?\s*)?results\s*$",
        r"^\s*(?:\d+\.?\s*)?discussion\s*$",
        r"^\s*(?:\d+\.?\s*)?conclusion[s]?\s*$",
        r"^\s*(?:\d+\.?\s*)?proposed\s+method(?:ology)?\s*$",
        r"^\s*(?:\d+\.?\s*)?method\s*$",
        r"^\s*(?:\d+\.?\s*)?references\s*$",
        r"^\s*(?:\d+\.?\s*)?bibliography\s*$"
    ]
    
    for line in lines:
        if not lr_started:
            for pattern in lr_headings:
                if re.match(pattern, line, re.IGNORECASE):
                    lr_started = True
                    break
        else:
            is_end = False
            for pattern in end_headings:
                if re.match(pattern, line, re.IGNORECASE):
                    is_end = True
                    break
            if is_end:
                break
            lr_lines.append(line)
            
    if not lr_started:
        return text
    return '\n'.join(lr_lines)
    
def extract_bibliography_section(text: str) -> str:
    lines = text.split('\n')
    bib_started = False
    bib_lines = []
    
    bib_headings = [
        r"^\s*(?:\d+\.?\s*)?references\s*$",
        r"^\s*(?:\d+\.?\s*)?bibliography\s*$"
    ]
    
    for line in lines:
        if not bib_started:
            for pattern in bib_headings:
                if re.match(pattern, line, re.IGNORECASE):
                    bib_started = True
                    break
        else:
            bib_lines.append(line)
            
    if not bib_started:
        return ""
    return '\n'.join(bib_lines)

text = """
1. Introduction
2. Related Work
This approach was proposed in [1].
We also use models from Smith et al. (2024).
3. Methodology
4. References
[1] Manakul, P., Liusie, A., Gales, M.J.F.: SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models (2023).
Smith, J., Jones, A.: Another Paper. (2024).
"""

lr_text = extract_literature_review_section(text)
print("LR Text:", lr_text.strip())

bib_text = extract_bibliography_section(text)
print("Bib Text:", bib_text.strip())

refs = parse_bibliography_text(bib_text)
print("Parsed Refs:")
for r in refs:
    print(r)
