import re

# Test title matching logic
def normalize_text(text: str) -> str:
    if not text: return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def calc_title_sim(title: str, ref_text: str) -> float:
    t_norm = normalize_text(title)
    r_norm = normalize_text(ref_text)
    
    if not t_norm or not r_norm: return 0.0
    
    # If the exact normalized title is in the normalized ref text, it's a huge match.
    if t_norm in r_norm:
        # Penalize slightly if the title is super short, but generally it's a match.
        return 1.0 if len(t_norm.split()) > 2 else 0.9
        
    t_tokens = set(t_norm.split())
    r_tokens = set(r_norm.split())
    if not t_tokens: return 0.0
    
    overlap = len(t_tokens.intersection(r_tokens)) / len(t_tokens)
    return overlap

print("Sim 1:", calc_title_sim("Attention is all you need", "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L. & Polosukhin, I. Attention is all you need. Advances in neural information processing systems, 30 (2017)."))
print("Sim 2:", calc_title_sim("Large language models meet computer vision: A brief survey", "Hamadi, R.: Large language models meet computer vision: A brief survey (2023) arXiv:2311.16673"))
