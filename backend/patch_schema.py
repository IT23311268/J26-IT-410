import re

schema_path = "models/schemas.py"
with open(schema_path, "r", encoding="utf-8") as f:
    content = f.read()

new_detail = """class ReferenceVerificationDetail(BaseModel):
    status: str
    reason: str
    match: Optional[dict] = None
    score: float
    verification_detail: Optional[str] = None
    matched_source: Optional[str] = None
    matched_title: Optional[str] = None
    matched_authors: Optional[str] = None
    matched_year: Optional[str] = None
    matched_doi: Optional[str] = None
    matched_arxiv: Optional[str] = None
    confidence: Optional[float] = None"""

content = re.sub(
    r"class ReferenceVerificationDetail\(BaseModel\):\n    status: str\n    reason: str\n    match: Optional\[dict\] = None\n    score: float",
    new_detail,
    content
)

with open(schema_path, "w", encoding="utf-8") as f:
    f.write(content)
